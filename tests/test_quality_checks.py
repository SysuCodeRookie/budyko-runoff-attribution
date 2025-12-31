"""
质量检查模块测试

测试validation模块中的QualityChecker类和数值稳定性工具函数。

作者: Budyko归因分析系统开发团队
日期: 2025-01-01
"""

import pytest
import numpy as np
from src.validation.quality_checks import (
    QualityChecker,
    safe_divide,
    safe_log,
    safe_power
)


class TestWaterBalanceCheck:
    """测试水量平衡检查"""
    
    def test_perfect_balance(self):
        """测试完美水量平衡"""
        is_valid, msg = QualityChecker.check_water_balance(
            P=1000, Q=400, E=600, tolerance=0.1
        )
        assert is_valid is True
        assert "通过检验" in msg
    
    def test_small_imbalance(self):
        """测试小幅失衡（在容忍范围内）"""
        is_valid, msg = QualityChecker.check_water_balance(
            P=1000, Q=400, E=590, tolerance=0.1
        )
        assert is_valid is True
    
    def test_large_imbalance(self):
        """测试大幅失衡（超出容忍范围）"""
        is_valid, msg = QualityChecker.check_water_balance(
            P=1000, Q=400, E=500, tolerance=0.1
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_negative_P(self):
        """测试负降水量"""
        is_valid, msg = QualityChecker.check_water_balance(
            P=-100, Q=50, E=50, tolerance=0.1
        )
        assert is_valid is False
        assert "错误" in msg


class TestRunoffRatioCheck:
    """测试径流系数检查"""
    
    def test_normal_ratio(self):
        """测试正常径流系数"""
        is_valid, msg = QualityChecker.check_runoff_ratio(
            Q=300, P=1000, min_ratio=0.05, max_ratio=0.95
        )
        assert is_valid is True
        assert "通过检验" in msg
    
    def test_too_low_ratio(self):
        """测试过低径流系数"""
        is_valid, msg = QualityChecker.check_runoff_ratio(
            Q=30, P=1000, min_ratio=0.05, max_ratio=0.95
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_too_high_ratio(self):
        """测试过高径流系数"""
        is_valid, msg = QualityChecker.check_runoff_ratio(
            Q=970, P=1000, min_ratio=0.05, max_ratio=0.95
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_ratio_exceeds_one(self):
        """测试径流超过降水（违反水量平衡）"""
        is_valid, msg = QualityChecker.check_runoff_ratio(
            Q=1100, P=1000, min_ratio=0.05, max_ratio=0.95
        )
        assert is_valid is False
        assert "错误" in msg
        assert "违反水量平衡" in msg
    
    def test_negative_runoff(self):
        """测试负径流"""
        is_valid, msg = QualityChecker.check_runoff_ratio(
            Q=-50, P=1000, min_ratio=0.05, max_ratio=0.95
        )
        assert is_valid is False
        assert "错误" in msg


class TestParameterRangeCheck:
    """测试Budyko参数范围检查"""
    
    def test_normal_parameter(self):
        """测试正常参数值"""
        is_valid, msg = QualityChecker.check_parameter_range(
            n=2.5, min_n=0.1, max_n=10.0
        )
        assert is_valid is True
        assert "通过检验" in msg
    
    def test_too_small_parameter(self):
        """测试过小参数"""
        is_valid, msg = QualityChecker.check_parameter_range(
            n=0.05, min_n=0.1, max_n=10.0
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_too_large_parameter(self):
        """测试过大参数"""
        is_valid, msg = QualityChecker.check_parameter_range(
            n=15.0, min_n=0.1, max_n=10.0
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_negative_parameter(self):
        """测试负参数"""
        is_valid, msg = QualityChecker.check_parameter_range(
            n=-1.0, min_n=0.1, max_n=10.0
        )
        assert is_valid is False
        assert "错误" in msg
    
    def test_nan_parameter(self):
        """测试NaN参数"""
        is_valid, msg = QualityChecker.check_parameter_range(
            n=np.nan, min_n=0.1, max_n=10.0
        )
        assert is_valid is False
        assert "非有限值" in msg


class TestElasticitySignsCheck:
    """测试弹性系数符号检查"""
    
    def test_correct_signs(self):
        """测试正确的符号"""
        is_valid, msg = QualityChecker.check_elasticity_signs(
            eps_P=2.0, eps_PET=-1.0, eps_n=-0.5
        )
        assert is_valid is True
        assert "✓" in msg or "通过" in msg
    
    def test_wrong_eps_P(self):
        """测试错误的εP符号"""
        is_valid, msg = QualityChecker.check_elasticity_signs(
            eps_P=-0.5, eps_PET=-1.0, eps_n=-0.5
        )
        assert is_valid is False
        assert "εP" in msg
        assert "违反物理意义" in msg
    
    def test_wrong_eps_PET(self):
        """测试错误的εPET符号"""
        is_valid, msg = QualityChecker.check_elasticity_signs(
            eps_P=2.0, eps_PET=1.0, eps_n=-0.5
        )
        assert is_valid is False
        assert "εPET" in msg
    
    def test_wrong_eps_n(self):
        """测试错误的εn符号"""
        is_valid, msg = QualityChecker.check_elasticity_signs(
            eps_P=2.0, eps_PET=-1.0, eps_n=0.5
        )
        assert is_valid is False
        assert "εn" in msg
    
    def test_nan_elasticity(self):
        """测试NaN弹性系数"""
        is_valid, msg = QualityChecker.check_elasticity_signs(
            eps_P=np.nan, eps_PET=-1.0, eps_n=-0.5
        )
        assert is_valid is False
        assert "非有限值" in msg


class TestElasticitySumCheck:
    """测试弹性系数和检查"""
    
    def test_normal_sum(self):
        """测试正常的弹性系数和"""
        is_valid, msg = QualityChecker.check_elasticity_sum(
            eps_P=1.5, eps_PET=-0.5, min_sum=0.5, max_sum=1.5
        )
        assert is_valid is True
        assert "通过检验" in msg
    
    def test_sum_too_low(self):
        """测试弹性系数和过低"""
        is_valid, msg = QualityChecker.check_elasticity_sum(
            eps_P=0.8, eps_PET=-0.5, min_sum=0.5, max_sum=1.5
        )
        assert is_valid is False
        assert "警告" in msg
    
    def test_sum_too_high(self):
        """测试弹性系数和过高"""
        is_valid, msg = QualityChecker.check_elasticity_sum(
            eps_P=2.5, eps_PET=-0.8, min_sum=0.5, max_sum=1.5
        )
        assert is_valid is False
        assert "警告" in msg


class TestComprehensiveCheck:
    """测试综合质量检查"""
    
    def test_all_pass(self):
        """测试所有检查通过"""
        config = {
            'water_balance_tolerance': 0.1,
            'min_runoff_ratio': 0.05,
            'max_runoff_ratio': 0.95,
            'n_bounds': [0.1, 10.0],
            'elasticity_sum_range': [0.5, 1.5]
        }
        
        results = QualityChecker.comprehensive_check(
            P=1000, Q=400, E=600, n=2.5,
            eps_P=2.0, eps_PET=-1.0, eps_n=-0.5,
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) == 5  # 5项检查
        
        # 检查所有项是否通过
        for check_name, (is_valid, msg) in results.items():
            assert is_valid is True, f"{check_name} 未通过: {msg}"
    
    def test_some_fail(self):
        """测试部分检查失败"""
        config = {
            'water_balance_tolerance': 0.1,
            'min_runoff_ratio': 0.05,
            'max_runoff_ratio': 0.95,
            'n_bounds': [0.1, 10.0],
            'elasticity_sum_range': [0.5, 1.5]
        }
        
        results = QualityChecker.comprehensive_check(
            P=1000, Q=980, E=50, n=0.05,  # 异常数据
            eps_P=-0.5, eps_PET=1.0, eps_n=0.5,  # 错误符号
            config=config
        )
        
        # 应该有多项检查失败
        failed_checks = sum(1 for is_valid, _ in results.values() if not is_valid)
        assert failed_checks > 0


class TestNumericalStabilityFunctions:
    """测试数值稳定性工具函数"""
    
    def test_safe_divide_normal(self):
        """测试正常除法"""
        result = safe_divide(np.array([10, 20, 30]), np.array([2, 4, 5]))
        expected = np.array([5, 5, 6])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_divide_zero(self):
        """测试除零"""
        result = safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]))
        assert result[0] == 5.0
        assert np.isnan(result[1])
        assert result[2] == 6.0
    
    def test_safe_divide_custom_fill(self):
        """测试自定义填充值"""
        result = safe_divide(
            np.array([10, 20]), np.array([0, 0]), fill_value=-999
        )
        assert result[0] == -999
        assert result[1] == -999
    
    def test_safe_log_positive(self):
        """测试正数对数"""
        result = safe_log(np.array([1, 10, 100]))
        expected = np.log(np.array([1, 10, 100]))
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_log_zero(self):
        """测试零的对数"""
        result = safe_log(np.array([0]), epsilon=1e-10)
        assert result[0] == np.log(1e-10)
    
    def test_safe_log_negative(self):
        """测试负数对数"""
        result = safe_log(np.array([-1, -10]))
        assert np.isnan(result[0])
        assert np.isnan(result[1])
    
    def test_safe_power_positive(self):
        """测试正数幂运算"""
        result = safe_power(np.array([2, 3, 4]), 2.5)
        expected = np.power(np.array([2, 3, 4]), 2.5)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_power_zero(self):
        """测试零的幂运算"""
        result = safe_power(np.array([0]), 2.5, epsilon=1e-10)
        assert result[0] == (1e-10)**2.5
    
    def test_safe_power_negative(self):
        """测试负数幂运算"""
        result = safe_power(np.array([-1, -2]), 2.5)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
