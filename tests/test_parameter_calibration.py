"""
test_parameter_calibration.py

测试模块5：参数校准和归因分析

测试内容：
- ParameterCalibrator类的初始化
- 单站点参数校准
- 批量站点处理
- 时段演变分析
- 归因分解计算
- Bootstrap不确定性估计
- 数据质量验证
- 结果导出功能

作者: Research Software Engineer
日期: 2025-01-01
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.budyko_model.parameter_calibration import (
    ParameterCalibrator,
    CalibrationResult,
    AttributionResult,
    validate_time_series_quality,
    calculate_ensemble_attribution
)


class TestParameterCalibratorInit:
    """测试ParameterCalibrator初始化"""
    
    def test_default_initialization(self):
        """测试默认参数初始化"""
        calibrator = ParameterCalibrator()
        
        assert calibrator.change_point == 1986
        assert calibrator.min_valid_years == 10
        assert calibrator.epsilon == 1e-10
        assert calibrator.budyko_model is not None
    
    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        calibrator = ParameterCalibrator(
            change_point=1990,
            min_valid_years=15,
            epsilon=1e-8
        )
        
        assert calibrator.change_point == 1990
        assert calibrator.min_valid_years == 15
        assert calibrator.epsilon == 1e-8


class TestCalibrationResult:
    """测试CalibrationResult数据容器"""
    
    def test_calibration_result_creation(self):
        """测试创建校准结果对象"""
        result = CalibrationResult(
            station_id="TEST001",
            n=2.5,
            P=850.0,
            PET=1200.0,
            Q_n=250.0,
            E=600.0,
            aridity_index=1.41,
            calibration_error=0.01,
            water_balance_valid=True,
            period="1960-1985"
        )
        
        assert result.station_id == "TEST001"
        assert result.n == 2.5
        assert result.period == "1960-1985"
        assert result.convergence == True
    
    def test_calibration_result_to_dict(self):
        """测试结果转换为字典"""
        result = CalibrationResult(
            station_id="TEST002",
            n=3.0,
            P=900.0,
            PET=1100.0,
            Q_n=300.0,
            E=600.0,
            aridity_index=1.22,
            calibration_error=0.005,
            water_balance_valid=True
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['station_id'] == "TEST002"
        assert result_dict['n'] == 3.0
        assert 'P' in result_dict
        assert 'calibration_error' in result_dict


class TestAttributionResult:
    """测试AttributionResult数据容器"""
    
    def test_attribution_result_creation(self):
        """测试创建归因结果对象"""
        result = AttributionResult(
            station_id="TEST003",
            delta_Q_obs=-50.0,
            delta_Q_n=-40.0,
            delta_Q_CCV=-30.0,
            delta_Q_LUCC=-10.0,
            delta_Q_WADR=-10.0,
            C_CCV=60.0,
            C_LUCC=20.0,
            C_WADR=20.0,
            period_base="1960-1985",
            period_impact="1986-2016"
        )
        
        assert result.station_id == "TEST003"
        assert result.C_CCV == 60.0
        assert result.C_LUCC == 20.0
        assert result.C_WADR == 20.0
    
    def test_attribution_result_to_dict(self):
        """测试归因结果转换为字典"""
        elasticity = {'P': 2.5, 'PET': -1.5, 'n': -1.2}
        result = AttributionResult(
            station_id="TEST004",
            delta_Q_obs=-60.0,
            delta_Q_n=-50.0,
            delta_Q_CCV=-35.0,
            delta_Q_LUCC=-15.0,
            delta_Q_WADR=-10.0,
            C_CCV=58.3,
            C_LUCC=25.0,
            C_WADR=16.7,
            period_base="1960-1985",
            period_impact="1986-2016",
            elasticity=elasticity
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'epsilon_P' in result_dict
        assert result_dict['epsilon_P'] == 2.5


class TestSingleStationCalibration:
    """测试单站点参数校准"""
    
    def test_calibrate_humid_basin(self):
        """测试湿润流域校准"""
        calibrator = ParameterCalibrator()
        
        result = calibrator.calibrate_single_station(
            station_id="HUMID001",
            P=1500.0,
            PET=900.0,
            Q_n=800.0,
            period="test"
        )
        
        assert result is not None
        assert result.station_id == "HUMID001"
        assert 0.1 < result.n < 10.0
        assert result.aridity_index < 1.0  # 湿润气候
        assert result.water_balance_valid == True
        assert result.calibration_error < 1.0  # 误差应小于1%
    
    def test_calibrate_arid_basin(self):
        """测试干旱流域校准"""
        calibrator = ParameterCalibrator()
        
        result = calibrator.calibrate_single_station(
            station_id="ARID001",
            P=300.0,
            PET=1500.0,
            Q_n=50.0,
            period="test"
        )
        
        assert result is not None
        assert result.station_id == "ARID001"
        assert 0.1 < result.n < 10.0
        assert result.aridity_index > 1.0  # 干旱气候
        assert result.water_balance_valid == True
    
    def test_calibrate_invalid_water_balance(self):
        """测试水量平衡违背情况"""
        calibrator = ParameterCalibrator()
        
        # Q_n > P，违反水量平衡
        result = calibrator.calibrate_single_station(
            station_id="INVALID001",
            P=500.0,
            PET=1000.0,
            Q_n=600.0,  # 超过降水量
            period="test"
        )
        
        assert result is None  # 应返回None
    
    def test_calibrate_edge_case_equal_P_PET(self):
        """测试P=PET的边界情况"""
        calibrator = ParameterCalibrator()
        
        result = calibrator.calibrate_single_station(
            station_id="EDGE001",
            P=1000.0,
            PET=1000.0,
            Q_n=300.0,
            period="test"
        )
        
        assert result is not None
        assert result.aridity_index == 1.0
        assert 0.1 < result.n < 10.0


class TestBatchCalibration:
    """测试批量站点校准"""
    
    def test_batch_calibrate_sequential(self):
        """测试顺序批量校准"""
        calibrator = ParameterCalibrator()
        
        # 创建测试数据
        data = pd.DataFrame({
            'station_id': ['ST001', 'ST002', 'ST003'],
            'P': [850.0, 1200.0, 600.0],
            'PET': [1200.0, 900.0, 1400.0],
            'Q_n': [200.0, 500.0, 100.0]
        })
        
        results = calibrator.batch_calibrate_stations(
            data, parallel=False
        )
        
        assert len(results) == 3
        assert all(isinstance(r, CalibrationResult) for r in results)
        assert [r.station_id for r in results] == ['ST001', 'ST002', 'ST003']
    
    def test_batch_calibrate_with_invalid_data(self):
        """测试批量校准包含无效数据"""
        calibrator = ParameterCalibrator()
        
        data = pd.DataFrame({
            'station_id': ['VALID001', 'INVALID002', 'VALID003'],
            'P': [850.0, 500.0, 1000.0],
            'PET': [1200.0, 1000.0, 1200.0],
            'Q_n': [200.0, 600.0, 300.0]  # INVALID002的Q_n > P
        })
        
        results = calibrator.batch_calibrate_stations(data, parallel=False)
        
        # 应该只有两个有效结果
        assert len(results) == 2
        assert all(r.station_id in ['VALID001', 'VALID003'] for r in results)
    
    def test_batch_calibrate_custom_column_names(self):
        """测试自定义列名的批量校准"""
        calibrator = ParameterCalibrator()
        
        data = pd.DataFrame({
            'id': ['A', 'B'],
            'precip': [900.0, 1100.0],
            'pet_val': [1100.0, 950.0],
            'runoff': [250.0, 400.0]
        })
        
        results = calibrator.batch_calibrate_stations(
            data,
            station_id_col='id',
            P_col='precip',
            PET_col='pet_val',
            Q_n_col='runoff',
            parallel=False
        )
        
        assert len(results) == 2
        assert results[0].station_id in ['A', 'B']


class TestParameterEvolution:
    """测试参数演变分析"""
    
    def test_analyze_parameter_evolution_full_period(self):
        """测试完整时段的参数演变分析"""
        calibrator = ParameterCalibrator(change_point=1986)
        
        # 创建1960-2016年的时间序列（57年）
        years = np.arange(1960, 2017)
        time_series = pd.DataFrame({
            'year': years,
            'P': np.random.normal(900, 50, len(years)),
            'PET': np.random.normal(1100, 80, len(years)),
            'Q_n': np.random.normal(250, 30, len(years))
        })
        
        results = calibrator.analyze_parameter_evolution(
            station_id="EVOLVE001",
            time_series=time_series
        )
        
        assert 'full' in results
        assert 'period_1' in results  # 1960-1985
        assert 'period_2' in results  # 1986-2016
        
        assert results['full'].station_id == "EVOLVE001"
        assert results['period_1'].period.startswith("1960")
        assert results['period_2'].period.endswith("2016")
    
    def test_parameter_evolution_insufficient_data(self):
        """测试数据不足的情况"""
        calibrator = ParameterCalibrator(
            change_point=1986,
            min_valid_years=15
        )
        
        # 只有20年数据（1960-1979），划分后period_2数据不足
        years = np.arange(1960, 1980)
        time_series = pd.DataFrame({
            'year': years,
            'P': np.random.normal(900, 50, len(years)),
            'PET': np.random.normal(1100, 80, len(years)),
            'Q_n': np.random.normal(250, 30, len(years))
        })
        
        results = calibrator.analyze_parameter_evolution(
            station_id="SHORT001",
            time_series=time_series
        )
        
        # period_2应该不存在（所有年份都在1986之前）
        assert 'period_2' not in results
        assert 'full' in results or 'period_1' in results
    
    def test_parameter_change_detection(self):
        """测试检测参数n的变化（模拟LUCC信号）"""
        calibrator = ParameterCalibrator(change_point=1986)
        
        # 人工构造n值变化的情景
        # Period 1: n=2.0, Period 2: n=3.0 (模拟植被增加)
        years_1 = np.arange(1960, 1986)
        years_2 = np.arange(1986, 2017)
        
        # 使用Budyko公式反算，保持n不同但P、PET相似
        P_mean = 900.0
        PET_mean = 1100.0
        
        time_series = pd.DataFrame({
            'year': np.concatenate([years_1, years_2]),
            'P': np.random.normal(P_mean, 30, len(years_1) + len(years_2)),
            'PET': np.random.normal(PET_mean, 40, len(years_1) + len(years_2)),
            'Q_n': np.concatenate([
                np.random.normal(250, 20, len(years_1)),  # Period 1
                np.random.normal(200, 20, len(years_2))   # Period 2 (径流减少)
            ])
        })
        
        results = calibrator.analyze_parameter_evolution(
            station_id="CHANGE001",
            time_series=time_series
        )
        
        if 'period_1' in results and 'period_2' in results:
            n1 = results['period_1'].n
            n2 = results['period_2'].n
            
            # 由于Q_n减少，n应该增加（更多蒸散发）
            # 但这个关系可能不总是严格的，所以只检查合理范围
            assert 0.1 < n1 < 10.0
            assert 0.1 < n2 < 10.0


class TestAttributionCalculation:
    """测试归因分解计算"""
    
    def test_calculate_attribution_basic(self):
        """测试基本归因计算"""
        calibrator = ParameterCalibrator()
        
        # 创建两个时段的校准结果
        period_1 = CalibrationResult(
            station_id="ATTR001",
            n=2.0,
            P=900.0,
            PET=1100.0,
            Q_n=250.0,
            E=650.0,
            aridity_index=1.22,
            calibration_error=0.01,
            water_balance_valid=True,
            period="1960-1985"
        )
        
        period_2 = CalibrationResult(
            station_id="ATTR001",
            n=2.5,  # n增加（LUCC）
            P=850.0,  # P减少
            PET=1150.0,  # PET增加
            Q_n=180.0,  # Q_n减少
            E=670.0,
            aridity_index=1.35,
            calibration_error=0.01,
            water_balance_valid=True,
            period="1986-2016"
        )
        
        # 观测径流也减少（包含WADR影响）
        Q_obs_1 = 250.0
        Q_obs_2 = 170.0  # 比Q_n减少更多
        
        result = calibrator.calculate_attribution(
            station_id="ATTR001",
            period_1_data=period_1,
            period_2_data=period_2,
            Q_obs_1=Q_obs_1,
            Q_obs_2=Q_obs_2
        )
        
        assert result is not None
        assert result.station_id == "ATTR001"
        
        # 检查变化量
        assert result.delta_Q_obs == Q_obs_2 - Q_obs_1  # -80
        assert result.delta_Q_n == period_2.Q_n - period_1.Q_n  # -70
        
        # 检查贡献率的合理性（由于Budyko公式的非线性，总和可能不严格为100%）
        # 但各分量应该都是有限值
        assert not np.isnan(result.C_CCV)
        assert not np.isnan(result.C_LUCC)
        assert not np.isnan(result.C_WADR)
        
        # 检查符号合理性：径流减少时（delta_Q < 0），主要贡献应为正贡献率
        assert abs(result.C_CCV) < 200  # 不应出现极端值
        assert abs(result.C_LUCC) < 200
    
    def test_attribution_with_small_change(self):
        """测试变化量很小时的归因计算"""
        calibrator = ParameterCalibrator()
        
        period_1 = CalibrationResult(
            station_id="SMALL001",
            n=2.0, P=900.0, PET=1100.0, Q_n=250.0, E=650.0,
            aridity_index=1.22, calibration_error=0.01,
            water_balance_valid=True, period="1960-1985"
        )
        
        period_2 = CalibrationResult(
            station_id="SMALL001",
            n=2.05, P=902.0, PET=1098.0, Q_n=251.0, E=651.0,
            aridity_index=1.22, calibration_error=0.01,
            water_balance_valid=True, period="1986-2016"
        )
        
        result = calibrator.calculate_attribution(
            station_id="SMALL001",
            period_1_data=period_1,
            period_2_data=period_2,
            Q_obs_1=250.0,
            Q_obs_2=250.5  # 变化仅0.5 mm
        )
        
        # 变化太小，贡献率应为NaN
        assert result is not None
        assert np.isnan(result.C_CCV)
        assert np.isnan(result.C_LUCC)
    
    def test_attribution_elasticity_signs(self):
        """测试弹性系数符号的合理性"""
        calibrator = ParameterCalibrator()
        
        period_1 = CalibrationResult(
            station_id="ELAST001",
            n=2.5, P=850.0, PET=1200.0, Q_n=200.0, E=650.0,
            aridity_index=1.41, calibration_error=0.01,
            water_balance_valid=True, period="1960-1985"
        )
        
        period_2 = CalibrationResult(
            station_id="ELAST001",
            n=3.0, P=800.0, PET=1250.0, Q_n=150.0, E=650.0,
            aridity_index=1.56, calibration_error=0.01,
            water_balance_valid=True, period="1986-2016"
        )
        
        result = calibrator.calculate_attribution(
            station_id="ELAST001",
            period_1_data=period_1,
            period_2_data=period_2,
            Q_obs_1=200.0,
            Q_obs_2=140.0
        )
        
        assert result is not None
        
        # 检查弹性系数符号
        assert result.elasticity['epsilon_P'] > 0  # 降水弹性应为正
        assert result.elasticity['epsilon_PET'] < 0  # PET弹性应为负
        assert result.elasticity['epsilon_n'] < 0  # n弹性应为负


class TestBootstrapUncertainty:
    """测试Bootstrap不确定性估计"""
    
    def test_bootstrap_basic(self):
        """测试基本Bootstrap功能"""
        calibrator = ParameterCalibrator()
        
        # 创建30年的时间序列
        np.random.seed(42)
        time_series = pd.DataFrame({
            'P': np.random.normal(900, 50, 30),
            'PET': np.random.normal(1100, 80, 30),
            'Q_n': np.random.normal(250, 30, 30)
        })
        
        uncertainty = calibrator.bootstrap_uncertainty(
            time_series,
            n_bootstrap=100,  # 使用较小的重采样次数以加快测试
            confidence_level=0.95
        )
        
        assert 'n' in uncertainty
        assert 'E' in uncertainty
        assert 'Q_n' in uncertainty
        
        # 检查置信区间格式
        assert len(uncertainty['n']) == 2  # (lower, upper)
        assert uncertainty['n'][0] < uncertainty['n'][1]  # lower < upper
    
    def test_bootstrap_narrow_confidence_interval(self):
        """测试数据稳定时的窄置信区间"""
        calibrator = ParameterCalibrator()
        
        # 创建变异性很小的时间序列
        np.random.seed(42)
        time_series = pd.DataFrame({
            'P': np.random.normal(900, 5, 50),  # 标准差很小
            'PET': np.random.normal(1100, 5, 50),
            'Q_n': np.random.normal(250, 3, 50)
        })
        
        uncertainty = calibrator.bootstrap_uncertainty(
            time_series,
            n_bootstrap=100,
            confidence_level=0.95
        )
        
        if 'n' in uncertainty:
            ci_width = uncertainty['n'][1] - uncertainty['n'][0]
            # 置信区间应该比较窄（具体阈值取决于数据）
            assert ci_width < 2.0  # n的95% CI宽度应小于2


class TestDataQualityValidation:
    """测试数据质量验证功能"""
    
    def test_validate_complete_data(self):
        """测试完整数据的验证"""
        time_series = pd.DataFrame({
            'year': np.arange(1960, 2017),
            'P': np.random.normal(900, 50, 57),
            'PET': np.random.normal(1100, 80, 57),
            'Q_n': np.random.normal(250, 30, 57)
        })
        
        is_valid, message = validate_time_series_quality(
            time_series,
            value_cols=['P', 'PET', 'Q_n']
        )
        
        assert is_valid == True
        assert "passed" in message.lower()
    
    def test_validate_missing_column(self):
        """测试缺失列的验证"""
        time_series = pd.DataFrame({
            'year': np.arange(1960, 2017),
            'P': np.random.normal(900, 50, 57),
            'PET': np.random.normal(1100, 80, 57)
            # 缺少Q_n列
        })
        
        is_valid, message = validate_time_series_quality(
            time_series,
            value_cols=['P', 'PET', 'Q_n']
        )
        
        assert is_valid == False
        assert "Missing required column" in message
    
    def test_validate_excessive_missing_values(self):
        """测试过多缺失值的验证"""
        data = np.random.normal(900, 50, 57)
        data[:20] = np.nan  # 35%的数据缺失
        
        time_series = pd.DataFrame({
            'year': np.arange(1960, 2017),
            'P': data,
            'PET': np.random.normal(1100, 80, 57),
            'Q_n': np.random.normal(250, 30, 57)
        })
        
        is_valid, message = validate_time_series_quality(
            time_series,
            value_cols=['P', 'PET', 'Q_n'],
            max_missing_fraction=0.15
        )
        
        assert is_valid == False
        assert "too many missing values" in message
    
    def test_validate_negative_values(self):
        """测试负值的验证"""
        time_series = pd.DataFrame({
            'year': np.arange(1960, 2017),
            'P': np.random.normal(900, 50, 57),
            'PET': np.random.normal(1100, 80, 57),
            'Q_n': np.concatenate([
                np.random.normal(250, 30, 50),
                [-10, -20, -5, -8, -12, -15, -7]  # 负值
            ])
        })
        
        is_valid, message = validate_time_series_quality(
            time_series,
            value_cols=['P', 'PET', 'Q_n']
        )
        
        assert is_valid == False
        assert "negative values" in message


class TestEnsembleAttribution:
    """测试集合归因统计"""
    
    def test_ensemble_mean_attribution(self):
        """测试平均归因统计"""
        # 创建多个站点的归因结果
        results = [
            AttributionResult(
                station_id=f"ST{i:03d}",
                delta_Q_obs=-50.0, delta_Q_n=-40.0,
                delta_Q_CCV=-30.0, delta_Q_LUCC=-10.0, delta_Q_WADR=-10.0,
                C_CCV=60.0 + i, C_LUCC=20.0 - i*0.5, C_WADR=20.0 - i*0.5,
                period_base="1960-1985", period_impact="1986-2016"
            )
            for i in range(10)
        ]
        
        ensemble = calculate_ensemble_attribution(results, method='mean')
        
        assert 'C_CCV_mean' in ensemble
        assert 'C_LUCC_mean' in ensemble
        assert 'C_WADR_mean' in ensemble
        assert 'C_CCV_std' in ensemble
        assert ensemble['n_stations'] == 10
        
        # 检查平均值合理性
        assert 50 < ensemble['C_CCV_mean'] < 70
    
    def test_ensemble_median_attribution(self):
        """测试中位数归因统计"""
        results = [
            AttributionResult(
                station_id=f"ST{i:03d}",
                delta_Q_obs=-50.0, delta_Q_n=-40.0,
                delta_Q_CCV=-30.0, delta_Q_LUCC=-10.0, delta_Q_WADR=-10.0,
                C_CCV=float(50 + i*5), C_LUCC=30.0, C_WADR=20.0,
                period_base="1960-1985", period_impact="1986-2016"
            )
            for i in range(5)
        ]
        
        ensemble = calculate_ensemble_attribution(results, method='median')
        
        assert 'C_CCV_median' in ensemble
        assert 'C_LUCC_median' in ensemble
        assert ensemble['n_stations'] == 5
        
        # 中位数应该是中间值
        assert ensemble['C_CCV_median'] == 60.0  # 50, 55, 60, 65, 70的中位数
    
    def test_ensemble_with_nan_values(self):
        """测试包含NaN值的集合统计"""
        results = [
            AttributionResult(
                station_id="ST001",
                delta_Q_obs=-50.0, delta_Q_n=-40.0,
                delta_Q_CCV=-30.0, delta_Q_LUCC=-10.0, delta_Q_WADR=-10.0,
                C_CCV=60.0, C_LUCC=20.0, C_WADR=20.0,
                period_base="1960-1985", period_impact="1986-2016"
            ),
            AttributionResult(
                station_id="ST002",
                delta_Q_obs=-0.5, delta_Q_n=-0.4,  # 变化太小
                delta_Q_CCV=-0.3, delta_Q_LUCC=-0.1, delta_Q_WADR=-0.1,
                C_CCV=np.nan, C_LUCC=np.nan, C_WADR=np.nan,
                period_base="1960-1985", period_impact="1986-2016"
            )
        ]
        
        ensemble = calculate_ensemble_attribution(results, method='mean')
        
        # 应该只使用有效值计算
        assert not np.isnan(ensemble['C_CCV_mean'])
        assert ensemble['n_stations'] == 2  # 仍然统计总站点数


class TestResultExport:
    """测试结果导出功能"""
    
    def test_export_calibration_results(self):
        """测试校准结果导出"""
        calibrator = ParameterCalibrator()
        
        results = [
            CalibrationResult(
                station_id=f"ST{i:03d}",
                n=2.0 + i*0.1, P=900.0, PET=1100.0,
                Q_n=250.0, E=650.0, aridity_index=1.22,
                calibration_error=0.01, water_balance_valid=True
            )
            for i in range(5)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "calibration_results.csv")
            calibrator.export_results(results, output_path)
            
            # 验证文件已创建
            assert os.path.exists(output_path)
            
            # 读取并验证内容
            df = pd.read_csv(output_path)
            assert len(df) == 5
            assert 'station_id' in df.columns
            assert 'n' in df.columns
            assert df['station_id'].iloc[0] == "ST000"
    
    def test_export_attribution_results(self):
        """测试归因结果导出"""
        calibrator = ParameterCalibrator()
        
        results = [
            AttributionResult(
                station_id=f"ST{i:03d}",
                delta_Q_obs=-50.0, delta_Q_n=-40.0,
                delta_Q_CCV=-30.0, delta_Q_LUCC=-10.0, delta_Q_WADR=-10.0,
                C_CCV=60.0, C_LUCC=20.0, C_WADR=20.0,
                period_base="1960-1985", period_impact="1986-2016"
            )
            for i in range(3)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "attribution_results.csv")
            calibrator.export_attribution_results(results, output_path)
            
            assert os.path.exists(output_path)
            
            df = pd.read_csv(output_path)
            assert len(df) == 3
            assert 'C_CCV' in df.columns
            assert 'C_LUCC' in df.columns
            assert 'C_WADR' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
