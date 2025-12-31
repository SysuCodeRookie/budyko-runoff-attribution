"""
test_budyko_attribution.py

Budyko归因分析模块测试套件

测试覆盖：
1. BudykoAttribution类初始化
2. 时段划分功能
3. 数据质量验证
4. 完整6步归因流程
5. 批量处理功能
6. 边界条件与异常处理
7. 物理一致性验证

作者: Research Software Engineer
日期: 2025-01-01
"""

import pytest
import numpy as np
import pandas as pd
import warnings

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.budyko_model.budyko_attribution import (
    BudykoAttribution,
    batch_attribution,
    summarize_attribution
)


# ========================================================================
# 测试数据生成
# ========================================================================

@pytest.fixture
def synthetic_station_data():
    """
    生成综合测试数据集
    
    模拟1960-2016年的水文数据，包含明显的突变特征：
    - 1960-1985（基准期）：稳定气候
    - 1986-2016（影响期）：降水减少、PET增加、n增大（退耕还林）
    """
    np.random.seed(42)
    years = np.arange(1960, 2017)
    n_years = len(years)
    
    # 基准期（1960-1985）
    pre_years = years < 1986
    post_years = years >= 1986
    
    # 降水：基准期800mm，影响期750mm（减少6.25%）
    P = np.where(pre_years, 
                 np.random.normal(800, 50, n_years),
                 np.random.normal(750, 50, n_years))
    
    # PET：基准期1000mm，影响期1100mm（增加10%）
    PET = np.where(pre_years,
                   np.random.normal(1000, 60, n_years),
                   np.random.normal(1100, 60, n_years))
    
    # 天然径流：通过Budyko方程计算（n基准期2.0，影响期2.5）
    n = np.where(pre_years, 2.0, 2.5)
    Qn = P - P * PET / (P**n + PET**n)**(1/n)
    
    # 观测径流：天然径流减去人类取用水（基准期5%，影响期15%）
    water_use_rate = np.where(pre_years, 0.05, 0.15)
    Qo = Qn * (1 - water_use_rate)
    
    data = pd.DataFrame({
        'year': years,
        'P': P,
        'PET': PET,
        'Qn': Qn,
        'Qo': Qo
    })
    
    return data


@pytest.fixture
def loess_plateau_data():
    """
    黄土高原典型流域数据（真实场景模拟）
    
    特征：
    - 半干旱气候
    - 1999年开始大规模退耕还林
    - 降水略有减少，n显著增大
    """
    np.random.seed(123)
    years = np.arange(1970, 2017)
    n_years = len(years)
    
    pre_years = years < 1999
    post_years = years >= 1999
    
    # 半干旱气候：P=450mm, PET=1200mm
    P = np.where(pre_years,
                 np.random.normal(450, 40, n_years),
                 np.random.normal(430, 40, n_years))  # 略减
    
    PET = np.where(pre_years,
                   np.random.normal(1200, 80, n_years),
                   np.random.normal(1250, 80, n_years))  # 增加
    
    # 退耕还林导致n大幅增加：1.5 → 2.8
    n = np.where(pre_years, 1.5, 2.8)
    Qn = P - P * PET / (P**n + PET**n)**(1/n)
    
    # 人类取用水影响较小（灌溉区较少）
    water_use_rate = 0.08
    Qo = Qn * (1 - water_use_rate)
    
    return pd.DataFrame({
        'year': years,
        'P': P,
        'PET': PET,
        'Qn': Qn,
        'Qo': Qo
    })


# ========================================================================
# 测试类1: 初始化与配置
# ========================================================================

class TestInitialization:
    """测试BudykoAttribution类的初始化"""
    
    def test_init_with_valid_data(self, synthetic_station_data):
        """测试正常初始化"""
        attribution = BudykoAttribution(synthetic_station_data)
        
        assert attribution.data is not None
        assert len(attribution.data) == 57  # 1960-2016
        assert attribution.pre_period is None  # 未设置时段
        assert attribution.post_period is None
    
    def test_init_missing_columns(self):
        """测试缺失必需列时的异常"""
        incomplete_data = pd.DataFrame({
            'year': [2000, 2001],
            'P': [800, 850]
            # 缺少PET, Qn, Qo
        })
        
        with pytest.raises(ValueError, match="缺少必需列"):
            BudykoAttribution(incomplete_data)
    
    def test_data_sorting(self):
        """测试数据自动排序"""
        unsorted_data = pd.DataFrame({
            'year': [2005, 2000, 2003],
            'P': [800, 850, 825],
            'PET': [1000, 950, 980],
            'Qn': [200, 220, 210],
            'Qo': [180, 200, 190]
        })
        
        attribution = BudykoAttribution(unsorted_data)
        assert list(attribution.data['year']) == [2000, 2003, 2005]


class TestPeriodSetting:
    """测试时段划分功能"""
    
    def test_set_periods_default(self, synthetic_station_data):
        """测试默认突变年份（1986）"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        assert attribution.change_year == 1986
        assert len(attribution.pre_period) == 26  # 1960-1985
        assert len(attribution.post_period) == 31  # 1986-2016
    
    def test_set_periods_custom(self, loess_plateau_data):
        """测试自定义突变年份"""
        attribution = BudykoAttribution(loess_plateau_data)
        attribution.set_periods(change_year=1999)
        
        assert attribution.change_year == 1999
        assert len(attribution.pre_period) == 29  # 1970-1998
        assert len(attribution.post_period) == 18  # 1999-2016
    
    def test_short_period_warning(self, synthetic_station_data):
        """测试时段过短时的警告"""
        attribution = BudykoAttribution(synthetic_station_data)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            attribution.set_periods(change_year=2014)  # 影响期仅3年
            
            assert len(w) == 1
            assert "过少" in str(w[0].message)


# ========================================================================
# 测试类2: 数据质量验证
# ========================================================================

class TestDataValidation:
    """测试数据质量检查功能"""
    
    def test_validate_good_data(self, synthetic_station_data):
        """测试正常数据通过验证"""
        attribution = BudykoAttribution(synthetic_station_data)
        is_valid, msg = attribution.validate_data_quality()
        
        assert is_valid
        assert "通过" in msg
    
    def test_detect_negative_values(self):
        """测试检测负值"""
        bad_data = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'P': [800, -50, 850],  # 负值
            'PET': [1000, 1020, 980],
            'Qn': [200, 220, 210],
            'Qo': [180, 200, 190]
        })
        
        attribution = BudykoAttribution(bad_data)
        is_valid, msg = attribution.validate_data_quality()
        
        assert not is_valid
        assert "负值" in msg
    
    def test_detect_water_balance_violation(self):
        """测试检测水量平衡违背（Qn > P）"""
        bad_data = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'P': [800, 850, 825],
            'PET': [1000, 1020, 980],
            'Qn': [200, 900, 210],  # 2001年Qn > P
            'Qo': [180, 200, 190]
        })
        
        attribution = BudykoAttribution(bad_data)
        is_valid, msg = attribution.validate_data_quality()
        
        assert not is_valid
        assert "违反水量平衡" in msg
    
    def test_warn_on_interbasin_transfer(self):
        """测试检测跨流域调水（Qo > Qn）"""
        transfer_data = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'P': [800, 850, 825],
            'PET': [1000, 1020, 980],
            'Qn': [200, 220, 210],
            'Qo': [180, 250, 190]  # 2001年Qo > Qn
        })
        
        attribution = BudykoAttribution(transfer_data)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            attribution.validate_data_quality()
            
            assert any("调水" in str(warning.message) for warning in w)


# ========================================================================
# 测试类3: 完整归因流程
# ========================================================================

class TestCompleteAttribution:
    """测试完整的6步归因流程"""
    
    def test_full_workflow(self, synthetic_station_data):
        """测试完整工作流程执行"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods(change_year=1986)
        
        results = attribution.run_attribution()
        
        # 验证所有必需键存在
        required_keys = [
            'n_full', 'n_pre', 'n_post',
            'delta_Qo', 'delta_Qn',
            'delta_Qn_CCV', 'delta_Qn_LUCC',
            'C_CCV', 'C_LUCC', 'C_WADR',
            'elasticity', 'validation'
        ]
        
        for key in required_keys:
            assert key in results, f"缺少键: {key}"
    
    def test_parameter_n_calibration(self, synthetic_station_data):
        """测试参数n率定结果合理性"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        results = attribution.run_attribution()
        
        # n应在合理范围内（0.5 ~ 10）
        assert 0.5 < results['n_full'] < 10
        assert 0.5 < results['n_pre'] < 10
        assert 0.5 < results['n_post'] < 10
        
        # 影响期n应大于基准期（模拟退耕还林）
        assert results['n_post'] > results['n_pre']
    
    def test_elasticity_calculation(self, synthetic_station_data):
        """测试弹性系数计算"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        results = attribution.run_attribution()
        elasticity = results['elasticity']
        
        # 验证符号
        assert elasticity['eps_P'] > 0, "εP应为正"
        assert elasticity['eps_PET'] < 0, "εPET应为负"
        assert elasticity['eps_n'] < 0, "εn应为负"
        
        # 验证数值范围
        assert 0.5 < elasticity['eps_P'] < 10
        assert -5 < elasticity['eps_PET'] < 0
        assert -3 < elasticity['eps_n'] < 0
    
    def test_contribution_sum(self, synthetic_station_data):
        """测试贡献率之和接近100%"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        results = attribution.run_attribution()
        
        contribution_sum = results['validation']['contribution_sum']
        
        # 贡献率之和应接近100%（允许±10%误差）
        assert 90 < contribution_sum < 110


class TestLoessPlateau:
    """测试黄土高原退耕还林案例"""
    
    def test_lucc_dominated(self, loess_plateau_data):
        """测试LUCC主导的归因结果"""
        attribution = BudykoAttribution(loess_plateau_data)
        attribution.set_periods(change_year=1999)
        
        results = attribution.run_attribution()
        
        # 黄土高原案例：LUCC应是主导因素
        assert abs(results['C_LUCC']) > abs(results['C_CCV']), \
            "退耕还林案例中，LUCC贡献应大于CCV"
        
        # n参数应显著增大
        assert results['delta_n'] > 0.5, \
            "退耕还林导致n显著增大"
        
        # 径流应减少
        assert results['delta_Qo'] < 0, \
            "退耕还林导致径流减少"


# ========================================================================
# 测试类4: 边界条件与异常处理
# ========================================================================

class TestEdgeCases:
    """测试边界条件和异常情况"""
    
    def test_no_periods_set(self, synthetic_station_data):
        """测试未设置时段时的异常"""
        attribution = BudykoAttribution(synthetic_station_data)
        # 未调用set_periods()
        
        with pytest.raises(ValueError, match="必须先调用"):
            attribution.run_attribution()
    
    def test_small_runoff_change(self):
        """测试径流变化过小时的警告"""
        # 构造变化极小的数据
        stable_data = pd.DataFrame({
            'year': np.arange(1970, 2017),
            'P': np.full(47, 800.0),
            'PET': np.full(47, 1000.0),
            'Qn': np.full(47, 200.0),
            'Qo': np.full(47, 180.0)  # 完全稳定
        })
        
        attribution = BudykoAttribution(stable_data)
        attribution.set_periods(change_year=1990)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = attribution.run_attribution()
            
            # 应发出径流变化过小的警告
            assert any("过小" in str(warning.message) for warning in w)
    
    def test_water_balance_violation_fails(self):
        """测试水量平衡违背导致率定失败"""
        bad_data = pd.DataFrame({
            'year': np.arange(1980, 2017),
            'P': np.full(37, 500.0),
            'PET': np.full(37, 1000.0),
            'Qn': np.full(37, 600.0),  # Qn > P，违背水量平衡
            'Qo': np.full(37, 500.0)
        })
        
        attribution = BudykoAttribution(bad_data)
        attribution.set_periods()
        
        with pytest.raises(ValueError, match="水量平衡"):
            attribution.run_attribution()


# ========================================================================
# 测试类5: 批量处理
# ========================================================================

class TestBatchAttribution:
    """测试批量归因功能"""
    
    @pytest.fixture
    def multi_station_data(self, synthetic_station_data, loess_plateau_data):
        """生成多站点数据"""
        # 为两个站点添加station_id
        data1 = synthetic_station_data.copy()
        data1['station_id'] = 'S001'
        
        data2 = loess_plateau_data.copy()
        data2['station_id'] = 'S002'
        
        # 合并
        multi_data = pd.concat([data1, data2], ignore_index=True)
        return multi_data
    
    def test_batch_processing(self, multi_station_data):
        """测试批量处理多个站点"""
        results = batch_attribution(
            multi_station_data,
            change_year=1986,
            station_id_col='station_id'
        )
        
        # 应返回2个站点的结果
        assert len(results) == 2
        assert 'S001' in results['station_id'].values
        assert 'S002' in results['station_id'].values
    
    def test_batch_with_failures(self, synthetic_station_data):
        """测试批量处理时部分站点失败"""
        # 使用已验证的好数据
        good_data = synthetic_station_data.copy()
        good_data['station_id'] = 'GOOD'
        
        # 创建坏数据
        bad_data = pd.DataFrame({
            'station_id': ['BAD'] * 30,
            'year': np.arange(1987, 2017),
            'P': np.full(30, 500.0),
            'PET': np.full(30, 1000.0),
            'Qn': np.full(30, 600.0),  # Qn > P违背水量平衡
            'Qo': np.full(30, 500.0)
        })
        
        mixed_data = pd.concat([good_data, bad_data], ignore_index=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = batch_attribution(mixed_data)
            
            # 应只返回好数据站点的结果
            assert len(results) == 1
            assert results['station_id'].iloc[0] == 'GOOD'
            
            # 应发出失败警告
            assert any("归因失败" in str(warning.message) for warning in w)


# ========================================================================
# 测试类6: 辅助功能
# ========================================================================

class TestUtilityFunctions:
    """测试辅助函数"""
    
    def test_summarize_attribution(self, synthetic_station_data):
        """测试结果摘要生成"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        results = attribution.run_attribution()
        summary = summarize_attribution(results)
        
        # 摘要应包含关键信息
        assert "参数n变化" in summary
        assert "气候要素变化" in summary
        assert "归因结果" in summary
        assert "主导因素" in summary
        
        # 应包含具体数值
        assert str(round(results['C_CCV'], 1)) in summary
        assert str(round(results['C_LUCC'], 1)) in summary


# ========================================================================
# 测试类7: 物理一致性验证
# ========================================================================

class TestPhysicalConsistency:
    """测试物理一致性"""
    
    def test_runoff_change_direction(self, loess_plateau_data):
        """测试径流变化方向与驱动因子一致"""
        attribution = BudykoAttribution(loess_plateau_data)
        attribution.set_periods(change_year=1999)
        
        results = attribution.run_attribution()
        
        # 降水减少 + PET增加 + n增大 → 径流应减少
        assert results['delta_Qo'] < 0, "综合作用应导致径流减少"
        
        # 每个因子的贡献符号应合理
        # 降水减少 → 径流减少（负贡献）
        if results['delta_P'] < 0:
            assert results['delta_Qn_CCV'] < 0
        
        # n增大 → 径流减少（负贡献）
        if results['delta_n'] > 0:
            assert results['delta_Qn_LUCC'] < 0
    
    def test_elasticity_amplification(self, synthetic_station_data):
        """测试降水弹性放大效应"""
        attribution = BudykoAttribution(synthetic_station_data)
        attribution.set_periods()
        
        results = attribution.run_attribution()
        
        # 降水变化的相对量
        delta_P_relative = results['delta_P'] / results['P_pre']
        
        # 由降水导致的径流变化相对量
        delta_Q_from_P = (results['elasticity']['eps_P'] * 
                          results['Qn_full'] / results['P_full'] * 
                          results['delta_P'])
        delta_Q_from_P_relative = delta_Q_from_P / results['Qn_pre']
        
        # 弹性系数>1，应有放大效应
        if results['elasticity']['eps_P'] > 1:
            assert abs(delta_Q_from_P_relative) > abs(delta_P_relative), \
                "εP>1时应有放大效应"


# ========================================================================
# 测试类8: 数值稳定性
# ========================================================================

class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_extreme_arid_condition(self):
        """测试极端干旱条件"""
        arid_data = pd.DataFrame({
            'year': np.arange(1980, 2017),
            'P': np.random.normal(200, 30, 37),
            'PET': np.random.normal(1800, 100, 37),
            'Qn': np.random.normal(20, 5, 37),
            'Qo': np.random.normal(15, 5, 37)
        })
        
        attribution = BudykoAttribution(arid_data)
        attribution.set_periods(change_year=2000)
        
        # 应能完成计算，不抛出异常
        results = attribution.run_attribution()
        
        # 结果应为有限值
        assert np.isfinite(results['n_full'])
        assert np.isfinite(results['C_CCV'])
    
    def test_extreme_humid_condition(self):
        """测试极端湿润条件"""
        humid_data = pd.DataFrame({
            'year': np.arange(1980, 2017),
            'P': np.random.normal(2000, 150, 37),
            'PET': np.random.normal(600, 50, 37),
            'Qn': np.random.normal(1200, 100, 37),
            'Qo': np.random.normal(1100, 100, 37)
        })
        
        attribution = BudykoAttribution(humid_data)
        attribution.set_periods(change_year=2000)
        
        results = attribution.run_attribution()
        
        assert np.isfinite(results['n_full'])
        assert np.isfinite(results['C_CCV'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
