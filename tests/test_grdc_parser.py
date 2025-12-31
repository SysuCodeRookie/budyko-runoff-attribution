"""
GRDC解析器单元测试
================

测试GRDCParser类的各项功能
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.grdc_parser import GRDCParser, validate_runoff_ratio


# 创建模拟GRDC文件
MOCK_GRDC_CONTENT = """# Title:                 GRDC STATION DATA FILE
#                        --------------
# Format:                DOS-ASCII
# Field delimiter:       ;
# missing values are indicated by -999.000
#
# file generation date:  2023-01-01
#
# GRDC-No.:              6335020
# River:                 YANGTZE RIVER
# Station:               YICHANG
# Country:               CN
# Latitude (DD):         30.70
# Longitude (DD):        111.25
# Catchment area (km2):  1000000.00
# Altitude (m ASL):      45.00
# First date:            1960-01-01
# Last date:             2020-12-31
# No. of years:          61
# Last update:           2023-01-01
#
# DATA
# YYYY-MM-DD;   Value; Flag
1960-01-01;  15000.0; A
1960-01-02;  14500.0; A
1960-01-03;  -999.0; M
1960-01-04;  14200.0; A
1960-01-05;  14800.0; A
1960-01-06;  15200.0; A
1960-01-07;  15500.0; A
1960-01-08;  15800.0; A
1960-01-09;  16000.0; A
1960-01-10;  16200.0; A
"""


@pytest.fixture
def mock_grdc_file():
    """创建临时GRDC文件"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(MOCK_GRDC_CONTENT)
        temp_path = f.name
    
    yield temp_path
    
    # 清理
    Path(temp_path).unlink()


def test_grdc_parser_init(mock_grdc_file):
    """测试初始化"""
    parser = GRDCParser(mock_grdc_file)
    assert parser.file_path.exists()
    assert parser.metadata == {}
    assert parser.timeseries is None


def test_grdc_parser_init_nonexistent_file():
    """测试不存在的文件"""
    with pytest.raises(FileNotFoundError):
        GRDCParser("nonexistent_file.txt")


def test_parse_metadata(mock_grdc_file):
    """测试元数据解析"""
    parser = GRDCParser(mock_grdc_file)
    metadata = parser.parse_metadata()
    
    assert metadata['grdc_no'] == 6335020
    assert metadata['station'] == 'YICHANG'
    assert metadata['river'] == 'YANGTZE RIVER'
    assert metadata['country'] == 'CN'
    assert abs(metadata['latitude'] - 30.70) < 0.01
    assert abs(metadata['longitude'] - 111.25) < 0.01
    assert abs(metadata['area_km2'] - 1000000.0) < 0.01
    assert abs(metadata['altitude_m'] - 45.0) < 0.01


def test_read_timeseries(mock_grdc_file):
    """测试时间序列读取"""
    parser = GRDCParser(mock_grdc_file)
    df = parser.read_timeseries()
    
    # 检查数据框架结构
    assert isinstance(df, pd.DataFrame)
    assert 'discharge_m3s' in df.columns
    assert 'flag' in df.columns
    
    # 检查数据内容
    assert len(df) == 10  # 10天数据
    assert df['discharge_m3s'].iloc[0] == 15000.0
    assert np.isnan(df['discharge_m3s'].iloc[2])  # -999应转为NaN
    
    # 检查日期索引
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp('1960-01-01')


def test_convert_to_depth(mock_grdc_file):
    """测试单位转换（流量→水深）"""
    parser = GRDCParser(mock_grdc_file)
    parser.parse_metadata()
    parser.read_timeseries()
    
    df = parser.convert_to_depth()
    
    assert 'runoff_mm_year' in df.columns
    
    # 验证转换公式
    # mm/year = m³/s × 86400 × 365.25 / (area_km² × 1e6) × 1000
    expected = 15000.0 * 86400 * 365.25 / (1000000.0 * 1e6) * 1000
    assert abs(df['runoff_mm_year'].iloc[0] - expected) < 0.01


def test_convert_to_depth_manual_area(mock_grdc_file):
    """测试手动指定面积"""
    parser = GRDCParser(mock_grdc_file)
    parser.read_timeseries()
    
    df = parser.convert_to_depth(area_km2=500000.0)
    
    # 使用手动面积的预期结果
    expected = 15000.0 * 86400 * 365.25 / (500000.0 * 1e6) * 1000
    assert abs(df['runoff_mm_year'].iloc[0] - expected) < 0.01


def test_convert_to_depth_no_area(mock_grdc_file):
    """测试缺少面积信息时的错误处理"""
    parser = GRDCParser(mock_grdc_file)
    parser.read_timeseries()
    parser.metadata = {}  # 清空元数据
    
    with pytest.raises(ValueError, match="集水区面积未知"):
        parser.convert_to_depth()


def test_aggregate_to_annual(mock_grdc_file):
    """测试年度聚合"""
    parser = GRDCParser(mock_grdc_file)
    parser.read_timeseries()
    
    annual = parser.aggregate_to_annual()
    
    assert isinstance(annual, pd.DataFrame)
    assert 'year' in annual.columns
    assert 'discharge_m3s' in annual.columns
    assert 'data_count' in annual.columns
    
    # 只有1960年的数据
    assert len(annual) == 1
    assert annual['year'].iloc[0] == 1960
    assert annual['data_count'].iloc[0] == 9  # 9天有效数据（1天缺测）


def test_quality_filter(mock_grdc_file):
    """测试质量过滤"""
    parser = GRDCParser(mock_grdc_file)
    parser.read_timeseries()
    
    # 这个测试文件只有10天数据，应该被过滤掉
    with pytest.warns(UserWarning, match="移除了.*个缺测率过高的年份"):
        filtered = parser.quality_filter(max_missing_pct=15, min_valid_days=300)
    
    # 应该没有数据通过过滤
    assert len(filtered) == 0


def test_validate_runoff_ratio():
    """测试径流系数验证函数"""
    # 正常情况
    is_valid, msg = validate_runoff_ratio(400, 800)
    assert is_valid
    assert "0.500" in msg
    
    # 径流>降水
    is_valid, msg = validate_runoff_ratio(1000, 800)
    assert not is_valid
    assert "违反水量平衡" in msg
    
    # 负径流
    is_valid, msg = validate_runoff_ratio(-100, 800)
    assert not is_valid
    assert "为负" in msg
    
    # 极端干旱
    is_valid, msg = validate_runoff_ratio(5, 1000)
    assert is_valid
    assert "极低" in msg
    
    # 极端湿润
    is_valid, msg = validate_runoff_ratio(960, 1000)
    assert is_valid
    assert "极高" in msg


def test_get_full_timeseries(mock_grdc_file):
    """测试一站式数据获取方法"""
    parser = GRDCParser(mock_grdc_file)
    
    # 不转换单位
    df1 = parser.get_full_timeseries(convert_to_depth=False)
    assert 'discharge_m3s' in df1.columns
    assert 'runoff_mm_year' not in df1.columns
    
    # 转换单位
    df2 = parser.get_full_timeseries(convert_to_depth=True)
    assert 'runoff_mm_year' in df2.columns


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
