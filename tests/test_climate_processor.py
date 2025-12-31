"""
气候数据处理器单元测试
====================

测试ClimateDataProcessor类的各项功能
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import tempfile
from shapely.geometry import Polygon
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.climate_processor import (
    ClimateDataProcessor,
    validate_climate_data,
    calculate_aridity_index
)


# 创建模拟NetCDF数据
@pytest.fixture
def mock_netcdf_file():
    """创建模拟的ISIMIP NetCDF文件"""
    # 创建模拟数据
    time = pd.date_range('2000-01-01', '2000-12-31', freq='D')
    lat = np.arange(30, 35, 0.5)
    lon = np.arange(110, 115, 0.5)
    
    # 创建降水数据（kg m-2 s-1）
    np.random.seed(42)
    pr_data = np.random.uniform(0, 1e-5, (len(time), len(lat), len(lon)))
    
    # 创建DataArray
    da = xr.DataArray(
        pr_data,
        coords={
            'time': time,
            'lat': lat,
            'lon': lon
        },
        dims=['time', 'lat', 'lon'],
        attrs={
            'units': 'kg m-2 s-1',
            'long_name': 'Precipitation',
            'standard_name': 'precipitation_flux'
        }
    )
    
    # 创建Dataset
    ds = xr.Dataset({'pr': da})
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as f:
        temp_path = f.name
    
    ds.to_netcdf(temp_path)
    
    yield temp_path
    
    # 清理
    Path(temp_path).unlink()


@pytest.fixture
def mock_basin_shapefile():
    """创建模拟流域边界Shapefile"""
    # 创建一个简单的多边形（覆盖部分网格）
    polygon = Polygon([
        (111, 31),
        (113, 31),
        (113, 33),
        (111, 33),
        (111, 31)
    ])
    
    gdf = gpd.GeoDataFrame(
        {'name': ['Test Basin'], 'area': [10000]},
        geometry=[polygon],
        crs="EPSG:4326"
    )
    
    # 保存到临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "test_basin.shp"
        gdf.to_file(temp_path)
        
        yield str(temp_path)


def test_climate_processor_init(mock_netcdf_file):
    """测试初始化"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    
    assert processor.file_path.exists()
    assert processor.variable == 'pr'
    assert processor.data is None
    assert processor._crs == "EPSG:4326"


def test_climate_processor_init_nonexistent():
    """测试不存在的文件"""
    with pytest.raises(FileNotFoundError):
        ClimateDataProcessor("nonexistent.nc")


def test_load_data(mock_netcdf_file):
    """测试数据加载"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    data = processor.load_data()
    
    assert isinstance(data, xr.DataArray)
    assert 'time' in data.dims
    assert 'lat' in data.dims
    assert 'lon' in data.dims
    assert processor.metadata['variable'] == 'pr'
    assert processor.metadata['units'] == 'kg m-2 s-1'


def test_load_data_auto_detect_variable(mock_netcdf_file):
    """测试自动检测变量名"""
    processor = ClimateDataProcessor(mock_netcdf_file)  # 不指定variable
    data = processor.load_data()
    
    assert processor.variable == 'pr'


def test_load_data_with_time_slice(mock_netcdf_file):
    """测试时间切片"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    data = processor.load_data(time_slice=slice('2000-01', '2000-06'))
    
    assert len(data.time) < 365
    assert data.time[0].dt.month == 1
    assert data.time[-1].dt.month == 6


def test_convert_units_precipitation(mock_netcdf_file):
    """测试降水单位转换"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    # 原始单位: kg m-2 s-1
    original_value = float(processor.data.isel(time=0, lat=0, lon=0))
    
    # 转换单位
    converted = processor.convert_units()
    converted_value = float(converted.isel(time=0, lat=0, lon=0))
    
    # 验证转换: mm/year = kg m-2 s-1 × 86400 × 365.25
    expected = original_value * 86400 * 365.25
    assert abs(converted_value - expected) < 1e-6
    assert converted.attrs['units'] == 'mm year-1'


def test_convert_units_temperature():
    """测试温度单位转换"""
    # 创建温度数据（K）
    time = pd.date_range('2000-01-01', '2000-01-10', freq='D')
    tas_data = np.full((len(time), 5, 5), 300.0)  # 300K
    
    da = xr.DataArray(
        tas_data,
        coords={'time': time, 'lat': np.arange(30, 35), 'lon': np.arange(110, 115)},
        dims=['time', 'lat', 'lon'],
        attrs={'units': 'K'}
    )
    
    processor = ClimateDataProcessor.__new__(ClimateDataProcessor)
    processor.variable = 'tas'
    processor.data = da
    
    # 转换
    converted = processor.convert_units()
    
    # 验证: °C = K - 273.15
    assert abs(float(converted.isel(time=0, lat=0, lon=0)) - 26.85) < 0.01
    assert converted.attrs['units'] == 'degC'


def test_aggregate_to_annual_sum(mock_netcdf_file):
    """测试年度聚合（求和）"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    # 先计算空间平均
    basin_mean = processor.data.mean(dim=['lat', 'lon'])
    
    # 聚合为年值
    annual = processor.aggregate_to_annual(basin_mean, method='sum')
    
    assert isinstance(annual, pd.Series)
    assert len(annual) == 1  # 只有2000年
    assert annual.index[0] == 2000
    assert annual.values[0] > 0


def test_aggregate_to_annual_mean(mock_netcdf_file):
    """测试年度聚合（平均）"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    basin_mean = processor.data.mean(dim=['lat', 'lon'])
    annual = processor.aggregate_to_annual(basin_mean, method='mean')
    
    assert isinstance(annual, pd.Series)
    assert annual.index[0] == 2000


def test_calculate_basin_mean_no_weight(mock_netcdf_file):
    """测试流域平均（无权重）"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    basin_mean = processor.calculate_basin_mean(processor.data, area_weighted=False)
    
    assert 'lat' not in basin_mean.dims
    assert 'lon' not in basin_mean.dims
    assert 'time' in basin_mean.dims


def test_calculate_basin_mean_weighted(mock_netcdf_file):
    """测试流域平均（面积加权）"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    basin_mean = processor.calculate_basin_mean(processor.data, area_weighted=True)
    
    assert 'lat' not in basin_mean.dims
    assert 'lon' not in basin_mean.dims


def test_validate_climate_data():
    """测试气候数据验证"""
    # 正常情况
    is_valid, msg = validate_climate_data(800, 1200, 400)
    assert is_valid
    
    # 径流大于降水
    is_valid, msg = validate_climate_data(800, 1200, 900)
    assert not is_valid
    assert "违反水量平衡" in msg
    
    # 负值
    is_valid, msg = validate_climate_data(-100, 1200, 400)
    assert not is_valid
    assert "负值" in msg
    
    # 极端值警告
    is_valid, msg = validate_climate_data(12000, 1200, 400)
    assert is_valid
    assert "极高" in msg


def test_calculate_aridity_index():
    """测试干旱指数计算"""
    # 湿润区
    ai, climate = calculate_aridity_index(1000, 150)
    assert ai == 0.15
    assert climate == "湿润"
    
    # 半干旱区
    ai, climate = calculate_aridity_index(500, 300)
    assert ai == 0.6
    assert climate == "半干旱"
    
    # 极端干旱区
    ai, climate = calculate_aridity_index(200, 800)
    assert ai == 4.0
    assert climate == "极端干旱"
    
    # 无效数据
    ai, climate = calculate_aridity_index(0, 800)
    assert np.isnan(ai)
    assert climate == "数据无效"


def test_extract_by_bbox(mock_netcdf_file, mock_basin_shapefile):
    """测试边界框提取"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    extracted = processor.extract_by_basin(mock_basin_shapefile, method='bbox')
    
    # 检查提取后的数据范围缩小
    assert len(extracted.lat) < len(processor.data.lat)
    assert len(extracted.lon) < len(processor.data.lon)


def test_get_resolution(mock_netcdf_file):
    """测试分辨率计算"""
    processor = ClimateDataProcessor(mock_netcdf_file, variable='pr')
    processor.load_data()
    
    lat_res, lon_res = processor._get_resolution(processor.data)
    
    assert abs(lat_res - 0.5) < 0.01
    assert abs(lon_res - 0.5) < 0.01


def test_isimip_units_mapping():
    """测试ISIMIP单位映射"""
    assert ClimateDataProcessor.ISIMIP_UNITS['pr'] == 'kg m-2 s-1'
    assert ClimateDataProcessor.ISIMIP_UNITS['tas'] == 'K'
    assert ClimateDataProcessor.TARGET_UNITS['pr'] == 'mm year-1'
    assert ClimateDataProcessor.TARGET_UNITS['tas'] == 'degC'


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
