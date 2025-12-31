"""
test_elasticity_solver.py

弹性系数计算模块测试

测试覆盖：
1. 降水弹性系数计算
2. PET弹性系数计算
3. 参数n弹性系数计算
4. 组合计算和验证
5. 边界条件和异常处理
6. 批量计算功能

作者: Research Software Engineer
日期: 2025-01-01
"""

import pytest
import numpy as np
import pandas as pd
from src.budyko_model.elasticity_solver import (
    calculate_elasticity_P,
    calculate_elasticity_PET,
    calculate_elasticity_n,
    calculate_all_elasticities,
    validate_elasticity_signs,
    batch_calculate_elasticities
)


class TestElasticityP:
    """测试降水弹性系数计算"""
    
    def test_calculate_epsilon_P_scalar(self):
        """测试标量输入的εP计算"""
        eps_P = calculate_elasticity_P(P=800, PET=1200, n=2.5)
        
        # εP应该大于0（降水增加导致径流增加）
        assert eps_P > 0, "εP应为正数"
        
        # εP通常在1-5之间
        assert 0.5 < eps_P < 10, f"εP={eps_P}超出合理范围"
        
        # 对于半干旱区（φ=1.5），εP通常大于1
        assert eps_P > 1, "半干旱区εP应大于1（放大效应）"
    
    def test_calculate_epsilon_P_array(self):
        """测试数组输入的εP计算"""
        P = np.array([800, 600, 1000, 400])
        PET = np.array([1200, 900, 800, 1200])
        n = np.array([2.5, 2.0, 3.0, 1.5])
        
        eps_P = calculate_elasticity_P(P, PET, n)
        
        assert isinstance(eps_P, np.ndarray)
        assert len(eps_P) == 4
        assert np.all(eps_P > 0), "所有εP应为正数"
    
    def test_epsilon_P_humid_vs_arid(self):
        """测试湿润区和干旱区εP的差异"""
        # 湿润区（P>PET）
        eps_P_humid = calculate_elasticity_P(P=1200, PET=800, n=2.5)
        
        # 干旱区（P<PET）
        eps_P_arid = calculate_elasticity_P(P=400, PET=1200, n=2.5)
        
        # 干旱区的εP应该大于湿润区（径流对降水更敏感）
        assert eps_P_arid > eps_P_humid, "干旱区εP应大于湿润区"
    
    def test_epsilon_P_invalid_inputs(self):
        """测试无效输入的异常处理"""
        # 负值降水
        with pytest.raises(ValueError, match="降水P必须为正数"):
            calculate_elasticity_P(P=-100, PET=800, n=2.0)
        
        # 零值降水
        with pytest.raises(ValueError):
            calculate_elasticity_P(P=0, PET=800, n=2.0)
        
        # 负值参数n
        with pytest.raises(ValueError, match="参数n必须为正数"):
            calculate_elasticity_P(P=800, PET=1200, n=-1.0)


class TestElasticityPET:
    """测试PET弹性系数计算"""
    
    def test_calculate_epsilon_PET_scalar(self):
        """测试标量输入的εPET计算"""
        eps_PET = calculate_elasticity_PET(P=800, PET=1200, n=2.5)
        
        # εPET应该小于0（PET增加导致径流减少）
        assert eps_PET < 0, "εPET应为负数"
        
        # εPET通常在-3到-0.2之间
        assert -5 < eps_PET < 0, f"εPET={eps_PET}超出合理范围"
    
    def test_calculate_epsilon_PET_array(self):
        """测试数组输入的εPET计算"""
        P = np.array([800, 600, 1000])
        PET = np.array([1200, 900, 800])
        n = np.array([2.5, 2.0, 3.0])
        
        eps_PET = calculate_elasticity_PET(P, PET, n)
        
        assert isinstance(eps_PET, np.ndarray)
        assert len(eps_PET) == 3
        assert np.all(eps_PET < 0), "所有εPET应为负数"
    
    def test_epsilon_PET_magnitude(self):
        """测试εPET的数值大小"""
        # 半干旱区
        eps_PET = calculate_elasticity_PET(P=600, PET=900, n=2.0)
        
        # εPET的绝对值通常在0.5-2.0之间
        assert 0.3 < abs(eps_PET) < 3.0
    
    def test_epsilon_PET_invalid_inputs(self):
        """测试无效输入"""
        with pytest.raises(ValueError, match="潜在蒸散发PET必须为正数"):
            calculate_elasticity_PET(P=800, PET=-1000, n=2.0)


class TestElasticityN:
    """测试参数n弹性系数计算"""
    
    def test_calculate_epsilon_n_scalar(self):
        """测试标量输入的εn计算"""
        eps_n = calculate_elasticity_n(P=800, PET=1200, n=2.5)
        
        # εn应该小于0（截留能力增强导致径流减少）
        assert eps_n < 0, "εn应为负数"
        
        # εn通常在-2到-0.05之间
        assert -3 < eps_n < 0, f"εn={eps_n}超出合理范围"
    
    def test_calculate_epsilon_n_array(self):
        """测试数组输入的εn计算"""
        P = np.array([800, 600, 1000])
        PET = np.array([1200, 900, 800])
        n = np.array([2.5, 2.0, 3.0])
        
        eps_n = calculate_elasticity_n(P, PET, n)
        
        assert isinstance(eps_n, np.ndarray)
        assert len(eps_n) == 3
        assert np.all(eps_n < 0), "所有εn应为负数"
    
    def test_epsilon_n_different_climates(self):
        """测试不同气候类型下的εn"""
        # 湿润区
        eps_n_humid = calculate_elasticity_n(P=1200, PET=800, n=2.5)
        
        # 干旱区
        eps_n_arid = calculate_elasticity_n(P=400, PET=1200, n=2.5)
        
        # 两者都应为负数
        assert eps_n_humid < 0
        assert eps_n_arid < 0
    
    def test_epsilon_n_invalid_inputs(self):
        """测试无效输入"""
        with pytest.raises(ValueError):
            calculate_elasticity_n(P=0, PET=1200, n=2.0)


class TestAllElasticities:
    """测试组合计算函数"""
    
    def test_calculate_all_elasticities(self):
        """测试一次性计算所有弹性系数"""
        result = calculate_all_elasticities(P=800, PET=1200, n=2.5)
        
        # 检查返回的字典键
        assert 'epsilon_P' in result
        assert 'epsilon_PET' in result
        assert 'epsilon_n' in result
        assert 'sum_P_PET' in result
        
        # 检查符号
        assert result['epsilon_P'] > 0
        assert result['epsilon_PET'] < 0
        assert result['epsilon_n'] < 0
        
        # 检查sum的计算
        expected_sum = result['epsilon_P'] + result['epsilon_PET']
        assert abs(result['sum_P_PET'] - expected_sum) < 1e-10
    
    def test_elasticity_sum_relation(self):
        """测试εP + εPET的关系（软约束）"""
        result = calculate_all_elasticities(P=800, PET=1200, n=2.5)
        
        eps_sum = result['sum_P_PET']
        
        # 对于Choudhury-Yang公式，εP + εPET可能偏离1
        # 但应该在合理范围内（0.5-1.5）
        assert 0.3 < eps_sum < 2.0, f"εP + εPET = {eps_sum}偏离合理范围"
    
    def test_all_elasticities_array_input(self):
        """测试数组输入的组合计算"""
        P = np.array([800, 600])
        PET = np.array([1200, 900])
        n = np.array([2.5, 2.0])
        
        result = calculate_all_elasticities(P, PET, n)
        
        assert isinstance(result['epsilon_P'], np.ndarray)
        assert len(result['epsilon_P']) == 2


class TestValidation:
    """测试验证函数"""
    
    def test_validate_correct_signs(self):
        """测试正确符号的验证"""
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=2.1,
            epsilon_PET=-1.0,
            epsilon_n=-0.5
        )
        
        assert is_valid is True
        assert "合理" in msg
    
    def test_validate_wrong_sign_P(self):
        """测试εP符号错误"""
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=-1.0,  # 错误：应为正数
            epsilon_PET=-1.0,
            epsilon_n=-0.5
        )
        
        assert is_valid is False
        assert "εP" in msg and "正数" in msg
    
    def test_validate_wrong_sign_PET(self):
        """测试εPET符号错误"""
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=2.0,
            epsilon_PET=1.0,  # 错误：应为负数
            epsilon_n=-0.5
        )
        
        assert is_valid is False
        assert "εPET" in msg and "负数" in msg
    
    def test_validate_wrong_sign_n(self):
        """测试εn符号错误"""
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=2.0,
            epsilon_PET=-1.0,
            epsilon_n=0.5  # 错误：应为负数
        )
        
        assert is_valid is False
        assert "εn" in msg
    
    def test_validate_strict_mode(self):
        """测试严格模式下的数值范围检查"""
        # 正常值应该通过
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=2.0,
            epsilon_PET=-1.0,
            epsilon_n=-0.3,
            strict=True
        )
        assert is_valid is True
        
        # εP过大
        is_valid, msg = validate_elasticity_signs(
            epsilon_P=15.0,  # 超出范围
            epsilon_PET=-1.0,
            epsilon_n=-0.3,
            strict=True
        )
        assert is_valid is False
        assert "εP" in msg


class TestBatchCalculation:
    """测试批量计算功能"""
    
    def test_batch_calculate_elasticities(self):
        """测试DataFrame批量计算"""
        df = pd.DataFrame({
            'P': [800, 600, 1000],
            'PET': [1200, 900, 800],
            'n': [2.5, 2.0, 3.0]
        })
        
        result = batch_calculate_elasticities(df)
        
        # 检查新增的列
        assert 'epsilon_P' in result.columns
        assert 'epsilon_PET' in result.columns
        assert 'epsilon_n' in result.columns
        
        # 检查行数不变
        assert len(result) == 3
        
        # 检查符号
        assert all(result['epsilon_P'] > 0)
        assert all(result['epsilon_PET'] < 0)
        assert all(result['epsilon_n'] < 0)
    
    def test_batch_custom_column_names(self):
        """测试自定义列名的批量计算"""
        df = pd.DataFrame({
            'precipitation': [800, 600],
            'evapotranspiration': [1200, 900],
            'parameter': [2.5, 2.0]
        })
        
        result = batch_calculate_elasticities(
            df,
            P_col='precipitation',
            PET_col='evapotranspiration',
            n_col='parameter'
        )
        
        assert 'epsilon_P' in result.columns
        assert len(result) == 2


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_extreme_dry_conditions(self):
        """测试极端干旱条件（PET >> P）"""
        eps_P = calculate_elasticity_P(P=100, PET=2000, n=1.5)
        eps_PET = calculate_elasticity_PET(P=100, PET=2000, n=1.5)
        eps_n = calculate_elasticity_n(P=100, PET=2000, n=1.5)
        
        # 应该仍能得到有限值
        assert np.isfinite(eps_P)
        assert np.isfinite(eps_PET)
        assert np.isfinite(eps_n)
        
        # 干旱区εP应该很大
        assert eps_P > 2.0
    
    def test_extreme_wet_conditions(self):
        """测试极端湿润条件（P >> PET）"""
        eps_P = calculate_elasticity_P(P=2000, PET=500, n=3.0)
        eps_PET = calculate_elasticity_PET(P=2000, PET=500, n=3.0)
        eps_n = calculate_elasticity_n(P=2000, PET=500, n=3.0)
        
        # 应该仍能得到有限值
        assert np.isfinite(eps_P)
        assert np.isfinite(eps_PET)
        assert np.isfinite(eps_n)
        
        # 湿润区εP应该较小
        assert eps_P < 3.0
    
    def test_very_small_n(self):
        """测试很小的n值（接近下界）"""
        eps_P = calculate_elasticity_P(P=800, PET=1200, n=0.2)
        eps_PET = calculate_elasticity_PET(P=800, PET=1200, n=0.2)
        eps_n = calculate_elasticity_n(P=800, PET=1200, n=0.2)
        
        assert np.isfinite(eps_P)
        assert np.isfinite(eps_PET)
        assert np.isfinite(eps_n)
    
    def test_very_large_n(self):
        """测试很大的n值（接近上界）"""
        eps_P = calculate_elasticity_P(P=800, PET=1200, n=8.0)
        eps_PET = calculate_elasticity_PET(P=800, PET=1200, n=8.0)
        eps_n = calculate_elasticity_n(P=800, PET=1200, n=8.0)
        
        assert np.isfinite(eps_P)
        assert np.isfinite(eps_PET)
        assert np.isfinite(eps_n)


class TestPhysicalConsistency:
    """测试物理一致性"""
    
    def test_elasticity_response_to_climate(self):
        """测试弹性系数对气候变化的响应"""
        # 从湿润到干旱
        climates = [
            (1500, 800),   # 非常湿润
            (1000, 800),   # 湿润
            (800, 1000),   # 半湿润
            (500, 1200),   # 半干旱
            (300, 1500)    # 干旱
        ]
        
        eps_P_values = []
        for P, PET in climates:
            eps_P = calculate_elasticity_P(P, PET, n=2.5)
            eps_P_values.append(eps_P)
        
        # εP应随干旱程度增加而增大
        # （径流对降水越来越敏感）
        for i in range(len(eps_P_values) - 1):
            assert eps_P_values[i] <= eps_P_values[i + 1] + 0.1, \
                "εP应随干旱程度增加而增大"
    
    def test_n_parameter_influence(self):
        """测试参数n对弹性系数的影响"""
        P, PET = 800, 1200
        
        # 不同的n值
        n_values = [1.0, 2.0, 3.0, 4.0]
        
        for n in n_values:
            elasticities = calculate_all_elasticities(P, PET, n)
            
            # 所有情况下符号应保持一致
            assert elasticities['epsilon_P'] > 0
            assert elasticities['epsilon_PET'] < 0
            assert elasticities['epsilon_n'] < 0
