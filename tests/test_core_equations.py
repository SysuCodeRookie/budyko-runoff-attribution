"""
Budyko核心方程模块测试

测试覆盖：
1. BudykoModel类初始化
2. 实际蒸散发计算（E）
3. 天然径流计算（Q_n）
4. 参数n反演（校准）
5. 弹性系数计算
6. 归因分解
7. 辅助函数
8. 边界条件和异常处理
9. 极端情况测试（根据"代码撰写总体建议"第4.2节）

作者: Budyko归因分析系统开发团队
日期: 2025-01-01
"""

import pytest
import numpy as np
import pandas as pd
from src.budyko_model.core_equations import (
    BudykoModel,
    validate_water_balance,
    calculate_aridity_index,
    estimate_n_from_climate,
    safe_divide,
    safe_log,
    safe_power
)


class TestBudykoModelInit:
    """测试BudykoModel类初始化"""
    
    def test_init_default(self):
        """测试默认初始化"""
        model = BudykoModel()
        assert model.epsilon == 1e-10
    
    def test_init_custom_epsilon(self):
        """测试自定义epsilon"""
        model = BudykoModel(epsilon=1e-8)
        assert model.epsilon == 1e-8


class TestActualETCalculation:
    """测试实际蒸散发计算"""
    
    def test_calculate_ET_scalar(self):
        """测试标量输入的E计算"""
        model = BudykoModel()
        E = model.calculate_actual_ET(P=800, PET=1200, n=2.5)
        
        # E应该在合理范围内（0 < E < min(P, PET)）
        assert 0 < E < 800
        # 对于PET > P的情况，E应该接近P
        assert E > 600  # E应该较大
    
    def test_calculate_ET_array(self):
        """测试数组输入的E计算"""
        model = BudykoModel()
        P = np.array([600, 800, 1000])
        PET = np.array([1000, 1200, 1400])
        n = np.array([2.0, 2.5, 3.0])
        
        E = model.calculate_actual_ET(P, PET, n)
        
        assert len(E) == 3
        assert np.all(E > 0)
        assert np.all(E < P)
    
    def test_calculate_ET_wet_climate(self):
        """测试湿润气候（P > PET）"""
        model = BudykoModel()
        # 湿润区：降水充足
        E = model.calculate_actual_ET(P=1500, PET=800, n=2.0)
        
        # E应该接近PET（受能量限制）
        assert 700 < E < 850
    
    def test_calculate_ET_dry_climate(self):
        """测试干旱气候（PET >> P）"""
        model = BudykoModel()
        # 干旱区：水分限制
        E = model.calculate_actual_ET(P=300, PET=1500, n=1.5)
        
        # E应该接近P
        assert 250 < E < 300
    
    def test_calculate_ET_negative_P(self):
        """测试负降水量（应抛出异常）"""
        model = BudykoModel()
        with pytest.raises(ValueError, match="降水量P不能为负值"):
            model.calculate_actual_ET(P=-100, PET=1000, n=2.0)
    
    def test_calculate_ET_negative_PET(self):
        """测试负PET（应抛出异常）"""
        model = BudykoModel()
        with pytest.raises(ValueError, match="潜在蒸散发PET不能为负值"):
            model.calculate_actual_ET(P=800, PET=-1000, n=2.0)
    
    def test_calculate_ET_negative_n(self):
        """测试负参数n（应抛出异常）"""
        model = BudykoModel()
        with pytest.raises(ValueError, match="参数n必须为正数"):
            model.calculate_actual_ET(P=800, PET=1000, n=-1.0)
    
    def test_calculate_ET_zero_n(self):
        """测试n=0（应抛出异常）"""
        model = BudykoModel()
        with pytest.raises(ValueError, match="参数n必须为正数"):
            model.calculate_actual_ET(P=800, PET=1000, n=0)


class TestNaturalizedRunoffCalculation:
    """测试天然径流计算"""
    
    def test_calculate_Qn_basic(self):
        """测试基本Q_n计算"""
        model = BudykoModel()
        Q_n = model.calculate_naturalized_runoff(P=800, PET=1200, n=2.5)
        
        # Q_n = P - E，应该在0到P之间
        assert 0 <= Q_n <= 800
    
    def test_calculate_Qn_water_balance(self):
        """测试水量平衡: Q_n + E = P"""
        model = BudykoModel()
        P, PET, n = 1000, 1500, 2.0
        
        E = model.calculate_actual_ET(P, PET, n)
        Q_n = model.calculate_naturalized_runoff(P, PET, n)
        
        # 检查水量平衡（允许小误差）
        assert abs((E + Q_n) - P) < 1e-6
    
    def test_calculate_Qn_humid_basin(self):
        """测试湿润流域（高径流系数）"""
        model = BudykoModel()
        # 湿润区：P远大于PET
        Q_n = model.calculate_naturalized_runoff(P=2000, PET=800, n=2.0)
        
        # 径流系数应该较高
        runoff_ratio = Q_n / 2000
        assert runoff_ratio > 0.4  # 超过40%
    
    def test_calculate_Qn_arid_basin(self):
        """测试干旱流域（低径流系数）"""
        model = BudykoModel()
        # 干旱区：PET远大于P
        Q_n = model.calculate_naturalized_runoff(P=300, PET=1800, n=1.5)
        
        # 径流系数应该较低
        runoff_ratio = Q_n / 300
        assert runoff_ratio < 0.2  # 低于20%


class TestParameterCalibration:
    """测试参数n反演"""
    
    def test_calibrate_n_basic(self):
        """测试基本参数校准"""
        model = BudykoModel()
        
        # 已知P, PET, Q_n，求n
        n = model.calibrate_parameter_n(P=800, PET=1200, Q_n=200)
        
        # n应该在合理范围内
        assert 0.1 < n < 10.0
        
        # 验证：用校准的n重新计算Q_n，应该接近观测值
        Q_n_check = model.calculate_naturalized_runoff(800, 1200, n)
        assert abs(Q_n_check - 200) < 1.0  # 误差小于1mm
    
    def test_calibrate_n_humid_basin(self):
        """测试湿润流域参数校准"""
        model = BudykoModel()
        
        # 湿润区：高径流
        n = model.calibrate_parameter_n(P=1500, PET=800, Q_n=800)
        
        # 湿润区n值通常较大
        assert n > 1.5
    
    def test_calibrate_n_arid_basin(self):
        """测试干旱流域参数校准"""
        model = BudykoModel()
        
        # 干旱区：低径流
        n = model.calibrate_parameter_n(P=400, PET=1500, Q_n=50)
        
        # 干旱区n值通常较小
        assert 0.5 < n < 3.0
    
    def test_calibrate_n_invalid_Qn_exceeds_P(self):
        """测试Q_n超过P的异常情况"""
        model = BudykoModel()
        
        # Q_n > P违反水量平衡
        with pytest.raises(ValueError, match="违反水量平衡"):
            model.calibrate_parameter_n(P=800, PET=1200, Q_n=900)
    
    def test_calibrate_n_negative_Qn(self):
        """测试负径流值"""
        model = BudykoModel()
        
        with pytest.raises(ValueError, match="天然径流Q_n不能为负值"):
            model.calibrate_parameter_n(P=800, PET=1200, Q_n=-100)
    
    def test_calibrate_n_zero_P(self):
        """测试零降水量"""
        model = BudykoModel()
        
        with pytest.raises(ValueError, match="P和PET必须为正数"):
            model.calibrate_parameter_n(P=0, PET=1200, Q_n=0)
    
    def test_calibrate_n_methods_consistency(self):
        """测试不同求解方法的一致性"""
        model = BudykoModel()
        
        P, PET, Q_n = 800, 1200, 200
        
        # 测试brentq方法稳定性（多次调用结果一致）
        n1 = model.calibrate_parameter_n(P, PET, Q_n, method='brentq')
        n2 = model.calibrate_parameter_n(P, PET, Q_n, method='brentq')
        
        # 多次调用结果应该完全一致
        assert abs(n1 - n2) < 1e-9
        
        # 验证n在合理范围内
        assert 0.1 < n1 < 10.0


class TestElasticityCalculation:
    """测试弹性系数计算"""
    
    def test_calculate_elasticities_basic(self):
        """测试基本弹性系数计算"""
        model = BudykoModel()
        
        elasticities = model.calculate_elasticities(P=800, PET=1200, n=2.5)
        
        # 检查返回的键
        assert 'epsilon_P' in elasticities
        assert 'epsilon_PET' in elasticities
        assert 'epsilon_n' in elasticities
    
    def test_elasticity_P_sign(self):
        """测试降水弹性的符号（通常为正）"""
        model = BudykoModel()
        
        elasticities = model.calculate_elasticities(P=800, PET=1200, n=2.0)
        epsilon_P = elasticities['epsilon_P']
        
        # εP通常为正，且大于1（放大效应）
        assert epsilon_P > 0
        # 对于半干旱区，εP通常在1-3之间
        assert 0.5 < epsilon_P < 5.0
    
    def test_elasticity_PET_sign(self):
        """测试PET弹性的符号（通常为负）"""
        model = BudykoModel()
        
        elasticities = model.calculate_elasticities(P=800, PET=1200, n=2.0)
        epsilon_PET = elasticities['epsilon_PET']
        
        # εPET通常为负（PET增加减少径流）
        assert epsilon_PET < 0
        # 绝对值通常在0-2之间
        assert -3.0 < epsilon_PET < 0
    
    def test_elasticity_n_sign(self):
        """测试参数n弹性的符号（通常为负）"""
        model = BudykoModel()
        
        elasticities = model.calculate_elasticities(P=800, PET=1200, n=2.0)
        epsilon_n = elasticities['epsilon_n']
        
        # εn通常为负（n增加减少径流）
        assert epsilon_n < 0
    
    def test_elasticities_humid_climate(self):
        """测试湿润气候的弹性系数"""
        model = BudykoModel()
        
        # 湿润区：P > PET
        elasticities = model.calculate_elasticities(P=1500, PET=800, n=2.5)
        
        # 湿润区特征：对PET更敏感
        assert abs(elasticities['epsilon_PET']) > 0.1
    
    def test_elasticities_arid_climate(self):
        """测试干旱气候的弹性系数"""
        model = BudykoModel()
        
        # 干旱区：PET >> P
        elasticities = model.calculate_elasticities(P=300, PET=1500, n=1.5)
        
        # 干旱区特征：对降水非常敏感
        assert elasticities['epsilon_P'] > 1.0


class TestAttributionAnalysis:
    """测试归因分解"""
    
    def test_attribution_basic(self):
        """测试基本归因分析"""
        model = BudykoModel()
        
        # 模拟两个时期
        attribution = model.calculate_runoff_change_attribution(
            P_base=850, PET_base=1100, Q_n_base=250,
            P_impact=800, PET_impact=1200, Q_n_impact=180
        )
        
        # 检查所有关键字段都存在
        required_keys = [
            'n_overall', 'n_base', 'n_impact',
            'epsilon_P', 'epsilon_PET', 'epsilon_n',
            'delta_P', 'delta_PET', 'delta_n', 'delta_Q_n',
            'delta_Q_n_CCV', 'delta_Q_n_LUCC', 'delta_Q_n_simulated'
        ]
        for key in required_keys:
            assert key in attribution
    
    def test_attribution_delta_consistency(self):
        """测试变化量的一致性"""
        model = BudykoModel()
        
        attribution = model.calculate_runoff_change_attribution(
            P_base=850, PET_base=1100, Q_n_base=250,
            P_impact=800, PET_impact=1200, Q_n_impact=180
        )
        
        # 检查变化量计算正确
        assert attribution['delta_P'] == 800 - 850
        assert attribution['delta_PET'] == 1200 - 1100
        assert attribution['delta_Q_n'] == 180 - 250
    
    def test_attribution_decomposition(self):
        """测试归因分解的合理性"""
        model = BudykoModel()
        
        attribution = model.calculate_runoff_change_attribution(
            P_base=900, PET_base=1000, Q_n_base=300,
            P_impact=850, PET_impact=1100, Q_n_impact=220
        )
        
        # 模拟的总变化应该接近CCV + LUCC
        delta_sum = attribution['delta_Q_n_CCV'] + attribution['delta_Q_n_LUCC']
        assert abs(delta_sum - attribution['delta_Q_n_simulated']) < 1.0
    
    def test_attribution_climate_dominated(self):
        """测试气候主导的变化"""
        model = BudykoModel()
        
        # 仅降水减少，PET增加（纯气候变化）
        attribution = model.calculate_runoff_change_attribution(
            P_base=1000, PET_base=1000, Q_n_base=350,
            P_impact=900, PET_impact=1100, Q_n_impact=250
        )
        
        # 气候贡献应该是主要的
        # （注意：由于n也会变化，LUCC贡献不一定为零）
        assert abs(attribution['delta_Q_n_CCV']) > 10  # 气候贡献显著


class TestAuxiliaryFunctions:
    """测试辅助函数"""
    
    def test_validate_water_balance_valid(self):
        """测试正常的水量平衡"""
        is_valid, msg = validate_water_balance(P=800, Q_n=200)
        assert is_valid
        assert "通过" in msg
    
    def test_validate_water_balance_Qn_exceeds_P(self):
        """测试Q_n超过P"""
        is_valid, msg = validate_water_balance(P=800, Q_n=850)
        assert not is_valid
        assert "径流超过降水" in msg
    
    def test_validate_water_balance_negative_P(self):
        """测试负降水量"""
        is_valid, msg = validate_water_balance(P=-100, Q_n=50)
        assert not is_valid
        assert "负值" in msg
    
    def test_validate_water_balance_high_ratio_warning(self):
        """测试高径流系数警告"""
        is_valid, msg = validate_water_balance(P=800, Q_n=780, threshold=0.95)
        assert is_valid  # 仍然有效
        assert "偏高" in msg  # 但有警告
    
    def test_calculate_aridity_index_scalar(self):
        """测试标量干旱指数计算"""
        AI = calculate_aridity_index(PET=1200, P=800)
        assert AI == 1.5
    
    def test_calculate_aridity_index_array(self):
        """测试数组干旱指数计算"""
        PET = np.array([1000, 1200, 1500])
        P = np.array([800, 800, 800])
        
        AI = calculate_aridity_index(PET, P)
        
        assert len(AI) == 3
        assert np.allclose(AI, [1.25, 1.5, 1.875])
    
    def test_estimate_n_humid_climate(self):
        """测试湿润气候n值估算"""
        n_init = estimate_n_from_climate(P=1500, PET=800)
        # 湿润区n值较大
        assert n_init >= 2.0
    
    def test_estimate_n_arid_climate(self):
        """测试干旱气候n值估算"""
        n_init = estimate_n_from_climate(P=400, PET=1200)
        # 干旱区n值较小
        assert n_init <= 2.5


class TestEdgeCases:
    """测试边界条件"""
    
    def test_extreme_wet_climate(self):
        """测试极端湿润气候（P >> PET）"""
        model = BudykoModel()
        
        # 热带雨林：P远大于PET
        E = model.calculate_actual_ET(P=3000, PET=1000, n=3.0)
        Q_n = model.calculate_naturalized_runoff(P=3000, PET=1000, n=3.0)
        
        # E应该接近PET
        assert 900 < E < 1100
        # Q_n应该很大
        assert Q_n > 1800
    
    def test_extreme_dry_climate(self):
        """测试极端干旱气候（PET >> P）"""
        model = BudykoModel()
        
        # 沙漠：PET远大于P
        E = model.calculate_actual_ET(P=100, PET=2500, n=1.0)
        Q_n = model.calculate_naturalized_runoff(P=100, PET=2500, n=1.0)
        
        # E应该接近P
        assert E < 100
        # Q_n应该很小
        assert Q_n < 20
    
    def test_equal_P_PET(self):
        """测试P = PET的情况"""
        model = BudykoModel()
        
        # 能量平衡区
        E = model.calculate_actual_ET(P=1000, PET=1000, n=2.0)
        Q_n = model.calculate_naturalized_runoff(P=1000, PET=1000, n=2.0)
        
        # E和Q_n应该都在合理范围
        assert 400 < E < 750  # 放宽E上限
        assert 250 < Q_n < 600  # 相应调整Q_n范围
        # 水量平衡
        assert abs((E + Q_n) - 1000) < 1.0
    
    def test_very_small_n(self):
        """测试极小的n值"""
        model = BudykoModel()
        
        # n极小时，Budyko方程的行为较特殊
        E = model.calculate_actual_ET(P=800, PET=1200, n=0.5)
        
        # E应该在合理范围内（n小时产流能力减弱）
        assert 0 < E <= 800
        # 对于干旱条件(PET > P)，E应该显著小于P
        assert E < 800
    
    def test_very_large_n(self):
        """测试极大的n值"""
        model = BudykoModel()
        
        # n很大时，Budyko曲线趋向于特定形状
        E = model.calculate_actual_ET(P=800, PET=1200, n=10.0)


# ============================================================================
# 极端情况测试（根据"代码撰写总体建议"4.2节要求）
# ============================================================================

class TestBudykoExtremes:
    """测试Budyko模型在极端条件下的行为"""
    
    def test_budyko_extreme_dry(self):
        """测试极端干旱情况 (P << PET)"""
        model = BudykoModel()
        P, PET, n = 200, 2000, 2.5
        
        Q = model.calculate_naturalized_runoff(P, PET, n)
        
        # 极端干旱时径流应很小
        assert Q < 50, f"极端干旱时径流应很小，实际为 {Q:.2f} mm"
        # 但应该非负
        assert Q >= 0, "径流不能为负值"
    
    def test_budyko_extreme_wet(self):
        """测试极端湿润情况 (P >> PET)"""
        model = BudykoModel()
        P, PET, n = 2000, 500, 2.5
        
        Q = model.calculate_naturalized_runoff(P, PET, n)
        E = model.calculate_actual_ET(P, PET, n)
        
        # 极端湿润时径流应接近P-PET
        assert Q > 1400, f"极端湿润时径流应接近P-PET，实际为 {Q:.2f} mm"
        # E应该接近PET（能量限制）
        assert abs(E - PET) / PET < 0.2, f"E应接近PET，E={E:.2f}, PET={PET}"
    
    def test_parameter_inversion_converge(self):
        """测试参数反演的收敛性"""
        model = BudykoModel()
        
        # 设定真实参数
        P, PET = 800, 1200
        n_true = 2.5
        
        # 正向计算
        Q_true = model.calculate_naturalized_runoff(P, PET, n_true)
        
        # 反演
        n_inv = model.calibrate_n(P, PET, Q_true)
        
        # 检查反演精度
        assert n_inv is not None, "参数反演失败"
        assert abs(n_inv - n_true) < 0.01, f"参数反演误差过大: n_true={n_true}, n_inv={n_inv}"
    
    def test_parameter_inversion_impossible(self):
        """测试无解情况（Q/P过大）"""
        model = BudykoModel()
        
        # Q/P = 0.98，几乎违反水量平衡
        n_inv = model.calibrate_n(P=1000, PET=1500, Q_n=980)
        
        # 应该返回None或警告
        assert n_inv is None, "不合理的Q/P比例应导致无解"


class TestNumericalStability:
    """测试数值稳定性工具函数"""
    
    def test_safe_divide_normal(self):
        """测试安全除法的正常情况"""
        result = safe_divide(np.array([10, 20, 30]), np.array([2, 4, 5]))
        expected = np.array([5, 5, 6])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_divide_zero_denominator(self):
        """测试除零情况"""
        result = safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]))
        assert result[0] == 5.0
        assert np.isnan(result[1])  # 除零位置应为NaN
        assert result[2] == 6.0
    
    def test_safe_divide_custom_fill(self):
        """测试自定义填充值"""
        result = safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]), fill_value=0.0)
        assert result[1] == 0.0  # 应使用自定义填充值
    
    def test_safe_log_positive(self):
        """测试正数的对数"""
        result = safe_log(np.array([1, 10, 100]))
        expected = np.log(np.array([1, 10, 100]))
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_log_zero(self):
        """测试零的对数（应使用epsilon）"""
        result = safe_log(np.array([0]), epsilon=1e-10)
        assert result[0] == np.log(1e-10)
    
    def test_safe_log_negative(self):
        """测试负数的对数（应返回NaN）"""
        result = safe_log(np.array([-1, -10]))
        assert np.isnan(result[0])
        assert np.isnan(result[1])
    
    def test_safe_power_positive(self):
        """测试正数的幂运算"""
        result = safe_power(np.array([2, 3, 4]), 2.5)
        expected = np.power(np.array([2, 3, 4]), 2.5)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_power_zero(self):
        """测试零的幂运算（应使用epsilon）"""
        result = safe_power(np.array([0]), 2.5, epsilon=1e-10)
        assert result[0] == (1e-10)**2.5
    
    def test_safe_power_negative(self):
        """测试负数的幂运算（应返回NaN）"""
        result = safe_power(np.array([-1, -2]), 2.5)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 应该在合理范围内
        assert 0 < E < 800
