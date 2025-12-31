"""
快速演示：GRDC解析器模块测试
"""
from src.data_preprocessing.grdc_parser import GRDCParser, validate_runoff_ratio
import tempfile
from pathlib import Path

print("="*70)
print("GRDC解析器模块 - 快速功能演示")
print("="*70)

# 创建模拟GRDC文件
mock_data = """# GRDC-No.:              1234567
# River:                 TEST RIVER
# Station:               TEST STATION
# Country:               CN
# Latitude (DD):         30.50
# Longitude (DD):        110.50
# Catchment area (km2):  50000.00
# Altitude (m ASL):      100.00
# First date:            2000-01-01
# Last date:             2000-12-31
# No. of years:          1
# DATA
2000-01-01;  1000.0; A
2000-01-02;  1100.0; A
2000-01-03;  1200.0; A
"""

# 创建临时文件
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
    f.write(mock_data)
    temp_file = f.name

try:
    print("\n1. 初始化解析器")
    parser = GRDCParser(temp_file)
    print("   ✓ 解析器创建成功")
    
    print("\n2. 提取元数据")
    metadata = parser.parse_metadata()
    print(f"   站点编号: {metadata['grdc_no']}")
    print(f"   河流名称: {metadata['river']}")
    print(f"   集水区面积: {metadata['area_km2']:,.0f} km²")
    print(f"   坐标: ({metadata['latitude']:.2f}°, {metadata['longitude']:.2f}°)")
    
    print("\n3. 读取时间序列")
    df = parser.read_timeseries()
    print(f"   数据点数: {len(df)}")
    print(f"   流量范围: {df['discharge_m3s'].min():.0f} - {df['discharge_m3s'].max():.0f} m³/s")
    
    print("\n4. 单位转换 (m³/s → mm/year)")
    df_depth = parser.convert_to_depth()
    print(f"   径流深度: {df_depth['runoff_mm_year'].mean():.2f} mm/year")
    
    print("\n5. 年度聚合")
    df_annual = parser.aggregate_to_annual()
    print(f"   年份: {df_annual['year'].iloc[0]}")
    print(f"   年均流量: {df_annual['discharge_m3s'].iloc[0]:.0f} m³/s")
    print(f"   有效天数: {df_annual['data_count'].iloc[0]}")
    
    print("\n6. 径流系数验证")
    is_valid, msg = validate_runoff_ratio(400, 800)
    print(f"   示例(400mm径流, 800mm降水): {msg}")
    print(f"   验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    
    print("\n" + "="*70)
    print("✓ 所有功能演示完成！模块工作正常")
    print("="*70)

finally:
    # 清理临时文件
    Path(temp_file).unlink()
    print("\n清理临时文件完成")
