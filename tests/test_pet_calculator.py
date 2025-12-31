"""
PET计算器单元测试
===============

测试PETCalculator类的各项功能。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.budyko_model.pet_calculator import (
    PETCalculator,
    convert_rsds_to_mj,
    estimate_missing_tmax_tmin,
    validate_pet_reasonableness,
    calculate_aridity_classification
)


class TestPETCalculatorInit:
    """测试PETCalculator初始化"""
    
    def test_init_valid(self):
        """测试正常初始化"""
        calc = PETCalculator(latitude=30.5, elevation=100)
        assert calc.latitude == 30.5
        assert calc.elevation == 100
        assert calc.method == 'pm'
    
    def test_init_invalid_latitude(self):
        """测试无效纬度"""
        with pytest.raises(ValueError, match="纬度必须在-90到90之间"):
            PETCalculator(latitude=100)
    
    def test_init_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValueError, match="不支持的方法"):
            PETCalculator(latitude=30, method='invalid')
    
    def test_init_extreme_elevation(self):
        """测试极端海拔高度（应发出警告）"""
        with pytest.warns(UserWarning, match="海拔高度异常"):
            PETCalculator(latitude=30, elevation=10000)


class TestPETCalculation:
    """测试PET计算功能"""
    
    def test_calculate_fao56_scalar(self):
        """测试单点PET计算"""
        calc = PETCalculator(latitude=30.5, elevation=50)
        
        pet = calc.calculate_fao56(
            tmean=25.0,
            tmax=30.0,
            tmin=20.0,
            rs=20.0,
            rh=60.0,
            uz=2.0
        )
        
        # PET应该是正值且在合理范围内
        assert isinstance(pet, float)
        assert 0 < pet < 20  # mm/day
    
    def test_calculate_fao56_series(self):
        """测试时间序列PET计算"""
        calc = PETCalculator(latitude=30.5)
        
        # 创建测试数据
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        n = len(dates)
        
        tmean = pd.Series(np.random.uniform(15, 25, n), index=dates)
        tmax = tmean + 5
        tmin = tmean - 5
        rs = pd.Series(np.random.uniform(15, 25, n), index=dates)
        
        pet = calc.calculate_fao56(
            tmean=tmean,
            tmax=tmax,
            tmin=tmin,
            rs=rs
        )
        
        assert isinstance(pet, pd.Series)
        assert len(pet) == n
        assert pet.min() > 0
        assert pet.max() < 20
    
    def test_calculate_with_defaults(self):
        """测试使用默认相对湿度和风速"""
        calc = PETCalculator(latitude=30.5)
        
        pet = calc.calculate_fao56(
            tmean=25.0,
            tmax=30.0,
            tmin=20.0,
            rs=20.0
            # rh和uz使用默认值
        )
        
        assert isinstance(pet, float)
        assert pet > 0
    
    def test_invalid_temperature_range(self):
        """测试无效的温度范围（tmax < tmin）"""
        calc = PETCalculator(latitude=30.5)
        
        with pytest.raises(ValueError, match="最高气温不能低于最低气温"):
            calc.calculate_fao56(
                tmean=25.0,
                tmax=20.0,  # 错误：tmax < tmin
                tmin=30.0,
                rs=20.0
            )
    
    def test_extreme_temperature_warning(self):
        """测试极端温度（应发出警告）"""
        calc = PETCalculator(latitude=30.5)
        
        with pytest.warns(UserWarning, match="气温超出合理范围"):
            calc.calculate_fao56(
                tmean=70.0,  # 异常高温
                tmax=75.0,
                tmin=65.0,
                rs=20.0
            )
    
    def test_invalid_humidity(self):
        """测试无效的相对湿度"""
        calc = PETCalculator(latitude=30.5)
        
        with pytest.raises(ValueError, match="相对湿度必须在0-100%之间"):
            calc.calculate_fao56(
                tmean=25.0,
                tmax=30.0,
                tmin=20.0,
                rs=20.0,
                rh=150.0  # 超过100%
            )


class TestHargreavesMethod:
    """测试Hargreaves简化方法"""
    
    def test_hargreaves_scalar(self):
        """测试Hargreaves单点计算"""
        calc = PETCalculator(latitude=30.5)
        
        pet = calc.calculate_hargreaves(
            tmean=25.0,
            tmax=30.0,
            tmin=20.0
        )
        
        assert isinstance(pet, float)
        assert pet > 0
    
    def test_hargreaves_series(self):
        """测试Hargreaves时间序列"""
        calc = PETCalculator(latitude=30.5)
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        tmean = pd.Series(np.linspace(20, 25, 10), index=dates)
        tmax = tmean + 5
        tmin = tmean - 5
        
        pet = calc.calculate_hargreaves(tmean, tmax, tmin)
        
        assert isinstance(pet, pd.Series)
        assert len(pet) == 10


class TestAggregation:
    """测试时间聚合功能"""
    
    def test_aggregate_to_annual_sum(self):
        """测试年度求和聚合"""
        calc = PETCalculator(latitude=30.5)
        
        # 创建2年的日数据
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        pet_daily = pd.Series(5.0, index=dates)  # 每天5mm
        
        pet_annual = calc.aggregate_to_annual(pet_daily, method='sum')
        
        assert len(pet_annual) == 2
        assert pet_annual.index[0] == 2020
        assert pet_annual.index[1] == 2021
        # 2020年是闰年（366天），2021年365天
        assert 1820 < pet_annual.iloc[0] < 1835  # 约5*366
        assert 1820 < pet_annual.iloc[1] < 1830  # 约5*365
    
    def test_aggregate_to_annual_mean(self):
        """测试年度平均聚合"""
        calc = PETCalculator(latitude=30.5)
        
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        pet_daily = pd.Series(5.0, index=dates)
        
        pet_annual = calc.aggregate_to_annual(pet_daily, method='mean')
        
        assert len(pet_annual) == 1
        # mean方法会乘以365.25
        assert abs(pet_annual.iloc[0] - 5.0 * 365.25) < 1


class TestClimateDataIntegration:
    """测试与climate_processor集成"""
    
    def test_calculate_from_climate_data(self):
        """测试从气候数据字典计算PET"""
        calc = PETCalculator(latitude=30.5)
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        climate_data = {
            'tas': pd.Series(25.0, index=dates),
            'tasmax': pd.Series(30.0, index=dates),
            'tasmin': pd.Series(20.0, index=dates),
            'rsds': pd.Series(20.0, index=dates),
            'hurs': pd.Series(60.0, index=dates),
            'sfcWind': pd.Series(2.0, index=dates)
        }
        
        pet = calc.calculate_from_climate_data(climate_data)
        
        assert isinstance(pet, pd.Series)
        assert len(pet) == 10
        assert pet.min() > 0
    
    def test_calculate_missing_variables(self):
        """测试缺少必需变量"""
        calc = PETCalculator(latitude=30.5)
        
        climate_data = {
            'tas': pd.Series([25.0]),
            # 缺少tasmax, tasmin, rsds
        }
        
        with pytest.raises(ValueError, match="缺少必需的气候变量"):
            calc.calculate_from_climate_data(climate_data)


class TestUtilityFunctions:
    """测试辅助函数"""
    
    def test_convert_rsds_to_mj_scalar(self):
        """测试单位转换（标量）"""
        rsds = 200.0  # W/m²
        rs = convert_rsds_to_mj(rsds)
        
        expected = 200.0 * 0.0864  # MJ/m²/day
        assert abs(rs - expected) < 0.001
    
    def test_convert_rsds_to_mj_series(self):
        """测试单位转换（Series）"""
        rsds = pd.Series([100, 200, 300])
        rs = convert_rsds_to_mj(rsds)
        
        assert isinstance(rs, pd.Series)
        assert len(rs) == 3
        assert abs(rs.iloc[0] - 100 * 0.0864) < 0.001
    
    def test_estimate_missing_tmax_tmin(self):
        """测试估算缺失的tmax/tmin"""
        tmean = pd.Series([20, 25, 30])
        
        with pytest.warns(UserWarning, match="使用估算的tmax/tmin"):
            tmax, tmin = estimate_missing_tmax_tmin(tmean, trange=10)
        
        assert isinstance(tmax, pd.Series)
        assert isinstance(tmin, pd.Series)
        assert all(tmax == tmean + 5)
        assert all(tmin == tmean - 5)
    
    def test_validate_pet_reasonableness_valid(self):
        """测试PET合理性检查（正常值）"""
        pet = 1500.0  # mm/year
        is_valid, msg = validate_pet_reasonableness(pet)
        
        assert is_valid
        assert "合理" in msg
    
    def test_validate_pet_reasonableness_negative(self):
        """测试PET合理性检查（负值）"""
        pet = -100.0
        is_valid, msg = validate_pet_reasonableness(pet)
        
        assert not is_valid
        assert "负值" in msg
    
    def test_validate_pet_reasonableness_extreme(self):
        """测试PET合理性检查（极端值）"""
        pet = 6000.0  # 超过5000
        is_valid, msg = validate_pet_reasonableness(pet)
        
        assert not is_valid
        assert "异常偏高" in msg
    
    def test_validate_pet_with_precipitation(self):
        """测试PET与降水的比值检查"""
        pet = 6000.0  # 极端高PET
        p = 100.0  # 很小的降水
        is_valid, msg = validate_pet_reasonableness(pet, p)
        
        # 干旱指数 = 6000/100 = 60，超过阈值50
        assert not is_valid
        assert "干旱指数" in msg
    
    def test_calculate_aridity_classification(self):
        """测试干旱指数分类"""
        # 湿润
        ai, climate = calculate_aridity_classification(pet=800, p=1000)
        assert ai < 1.0
        assert climate == "湿润"
        
        # 半湿润
        ai, climate = calculate_aridity_classification(pet=1200, p=1000)
        assert 1.0 <= ai < 2.0
        assert climate == "半湿润"
        
        # 半干旱
        ai, climate = calculate_aridity_classification(pet=3000, p=1000)
        assert 2.0 <= ai < 5.0
        assert climate == "半干旱"
        
        # 干旱
        ai, climate = calculate_aridity_classification(pet=6000, p=1000)
        assert ai >= 5.0
        assert climate == "干旱"


class TestEdgeCases:
    """测试边界情况"""
    
    def test_zero_radiation(self):
        """测试零辐射"""
        calc = PETCalculator(latitude=30.5)
        
        # 零辐射情况，PET应该很小
        pet = calc.calculate_fao56(
            tmean=25.0,
            tmax=30.0,
            tmin=20.0,
            rs=0.0  # 零辐射（夜晚）
        )
        
        # PET应该很小但非负
        assert pet >= 0
        assert pet < 2  # 在没有辐射的情况下PET主要来自气温差异
    
    def test_equator_latitude(self):
        """测试赤道纬度"""
        calc = PETCalculator(latitude=0.0)
        
        pet = calc.calculate_fao56(
            tmean=25.0,
            tmax=28.0,
            tmin=22.0,
            rs=20.0
        )
        
        assert pet > 0
    
    def test_polar_latitude(self):
        """测试极地纬度"""
        calc = PETCalculator(latitude=80.0)
        
        pet = calc.calculate_fao56(
            tmean=0.0,
            tmax=5.0,
            tmin=-5.0,
            rs=10.0
        )
        
        assert pet >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
