"""
气候数据处理器使用示例
===================

演示如何使用ClimateDataProcessor处理ISIMIP气候数据
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.climate_processor import (
    ClimateDataProcessor,
    validate_climate_data,
    calculate_aridity_index
)


def example_1_single_variable():
    """示例1: 处理单个变量（降水）"""
    print("=" * 70)
    print("示例1: 处理单个气候变量")
    print("=" * 70)
    
    # 替换为您的实际文件路径
    nc_file = "data/raw/ISIMIP/pr_GSWP3-W5E5_1960-2016.nc"
    basin_shp = "data/shapefiles/yangtze_basin.shp"
    
    try:
        print("\n1. 初始化处理器")
        processor = ClimateDataProcessor(nc_file, variable="pr")
        print(f"   ✓ 文件: {processor.file_path.name}")
        print(f"   ✓ 变量: {processor.variable}")
        
        print("\n2. 加载数据")
        data = processor.load_data(chunks={'time': 365})
        print(f"   数据维度: {data.dims}")
        print(f"   时间范围: {processor.metadata['time_range']}")
        print(f"   空间分辨率: {processor.metadata['spatial_resolution']}")
        
        print("\n3. 提取流域数据")
        basin_data = processor.extract_by_basin(basin_shp, method='clip')
        print(f"   提取后维度: {basin_data.dims}")
        
        print("\n4. 计算流域平均（面积加权）")
        basin_mean = processor.calculate_basin_mean(basin_data, area_weighted=True)
        print(f"   时间序列长度: {len(basin_mean)}")
        
        print("\n5. 单位转换")
        basin_mean = processor.convert_units(basin_mean)
        print(f"   转换后单位: {basin_mean.attrs.get('units')}")
        print(f"   均值: {float(basin_mean.mean()):.2f} mm/year")
        
        print("\n6. 聚合为年值")
        annual_pr = processor.aggregate_to_annual(basin_mean, method='sum')
        print(f"   年份数: {len(annual_pr)}")
        print(f"   年均降水: {annual_pr.mean():.2f} mm/year")
        
        return annual_pr
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {nc_file}")
        print("   请将文件路径替换为您的实际ISIMIP数据文件")
        return None


def example_2_complete_pipeline():
    """示例2: 使用完整流水线"""
    print("\n" + "=" * 70)
    print("示例2: 一站式处理流水线")
    print("=" * 70)
    
    nc_file = "data/raw/ISIMIP/tas_GSWP3-W5E5_1960-2016.nc"
    basin_shp = "data/shapefiles/yangtze_basin.shp"
    
    try:
        print("\n使用process_pipeline()方法一步完成所有处理:")
        
        processor = ClimateDataProcessor(nc_file, variable="tas")
        annual_tas = processor.process_pipeline(
            basin_geometry=basin_shp,
            convert_units=True,
            aggregate=True,
            aggregate_method='mean'  # 温度用平均值
        )
        
        print(f"\n结果:")
        print(f"  年均气温: {annual_tas.mean():.2f} °C")
        print(f"  最高年份: {annual_tas.idxmax()}, {annual_tas.max():.2f} °C")
        print(f"  最低年份: {annual_tas.idxmin()}, {annual_tas.min():.2f} °C")
        
        return annual_tas
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {nc_file}")
        return None


def example_3_batch_processing():
    """示例3: 批量处理多个变量"""
    print("\n" + "=" * 70)
    print("示例3: 批量处理多个气候变量")
    print("=" * 70)
    
    file_pattern = "data/raw/ISIMIP/{var}_GSWP3-W5E5_1960-2016.nc"
    variables = ['pr', 'tas', 'tasmax', 'tasmin', 'rsds', 'hurs', 'sfcWind']
    basin_shp = "data/shapefiles/yangtze_basin.shp"
    output_dir = "data/processed/climate/"
    
    try:
        results = ClimateDataProcessor.batch_process_variables(
            file_pattern=file_pattern,
            variables=variables,
            basin_geometry=basin_shp,
            output_dir=output_dir
        )
        
        print(f"\n✓ 成功处理 {len(results)} 个变量")
        print("\n各变量年均值:")
        for var, data in results.items():
            print(f"  {var:10s}: {data.mean():>10.2f}")
        
        return results
        
    except Exception as e:
        print(f"\n⚠️  批量处理失败: {e}")
        return None


def example_4_quality_checks():
    """示例4: 数据质量检查"""
    print("\n" + "=" * 70)
    print("示例4: 气候数据质量检查")
    print("=" * 70)
    
    # 模拟一些数据
    test_cases = [
        {'P': 800, 'PET': 1200, 'Q': 400, 'desc': '正常流域'},
        {'P': 1500, 'PET': 600, 'Q': 850, 'desc': '湿润流域'},
        {'P': 300, 'PET': 1500, 'Q': 50, 'desc': '干旱流域'},
        {'P': 800, 'PET': 1200, 'Q': 900, 'desc': '异常：Q>P'},
        {'P': -100, 'PET': 1200, 'Q': 400, 'desc': '异常：负降水'},
    ]
    
    print("\n水量平衡检查:")
    print(f"{'描述':<15} {'P(mm)':<8} {'PET(mm)':<8} {'Q(mm)':<8} {'状态':<6} {'说明'}")
    print("-" * 70)
    
    for case in test_cases:
        is_valid, msg = validate_climate_data(case['P'], case['PET'], case['Q'])
        status = "✓" if is_valid else "✗"
        print(f"{case['desc']:<15} {case['P']:<8.0f} {case['PET']:<8.0f} "
              f"{case['Q']:<8.0f} {status:<6} {msg}")


def example_5_aridity_classification():
    """示例5: 干旱指数与气候分类"""
    print("\n" + "=" * 70)
    print("示例5: 干旱指数计算与气候分类")
    print("=" * 70)
    
    basins = [
        {'name': '长江流域', 'P': 1100, 'PET': 900},
        {'name': '黄河流域', 'P': 450, 'PET': 1200},
        {'name': '塔里木河流域', 'P': 150, 'PET': 1800},
        {'name': '珠江流域', 'P': 1800, 'PET': 950},
    ]
    
    print("\n流域气候分类:")
    print(f"{'流域':<15} {'P(mm)':<8} {'PET(mm)':<10} {'干旱指数':<12} {'气候类型'}")
    print("-" * 70)
    
    for basin in basins:
        ai, climate = calculate_aridity_index(basin['P'], basin['PET'])
        print(f"{basin['name']:<15} {basin['P']:<8.0f} {basin['PET']:<10.0f} "
              f"{ai:<12.2f} {climate}")


def example_6_visualization():
    """示例6: 数据可视化"""
    print("\n" + "=" * 70)
    print("示例6: 气候数据可视化")
    print("=" * 70)
    
    # 生成模拟数据用于演示
    years = np.arange(1960, 2017)
    np.random.seed(42)
    
    # 模拟趋势 + 随机波动
    pr = 800 + np.random.normal(0, 100, len(years)) - 0.5 * (years - 1960)
    tas = 15 + 0.02 * (years - 1960) + np.random.normal(0, 0.5, len(years))
    
    annual_pr = pd.Series(pr, index=years, name='降水 (mm/year)')
    annual_tas = pd.Series(tas, index=years, name='气温 (°C)')
    
    print("\n绘制时间序列图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 子图1: 降水
    axes[0].plot(annual_pr.index, annual_pr.values, 
                marker='o', linewidth=2, markersize=4, color='steelblue')
    axes[0].axhline(y=annual_pr.mean(), color='red', linestyle='--', 
                   label=f'平均值: {annual_pr.mean():.1f} mm')
    axes[0].set_xlabel('年份', fontsize=12)
    axes[0].set_ylabel('年降水量 (mm)', fontsize=12)
    axes[0].set_title('长江流域年降水量变化（示例）', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 气温
    axes[1].plot(annual_tas.index, annual_tas.values, 
                marker='s', linewidth=2, markersize=4, color='orangered')
    
    # 添加趋势线
    z = np.polyfit(annual_tas.index, annual_tas.values, 1)
    p = np.poly1d(z)
    axes[1].plot(annual_tas.index, p(annual_tas.index), 
                linestyle='--', color='darkred', linewidth=2,
                label=f'趋势: {z[0]:.3f} °C/年')
    
    axes[1].set_xlabel('年份', fontsize=12)
    axes[1].set_ylabel('年均气温 (°C)', fontsize=12)
    axes[1].set_title('长江流域年均气温变化（示例）', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = "figures/climate_timeseries_example.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {output_file}")
    
    plt.show()


def example_7_water_year():
    """示例7: 水文年处理"""
    print("\n" + "=" * 70)
    print("示例7: 水文年 vs 日历年")
    print("=" * 70)
    
    nc_file = "data/raw/ISIMIP/pr_GSWP3-W5E5_1960-2016.nc"
    basin_shp = "data/shapefiles/yangtze_basin.shp"
    
    try:
        processor = ClimateDataProcessor(nc_file, variable="pr")
        processor.load_data()
        basin_data = processor.extract_by_basin(basin_shp)
        basin_mean = processor.calculate_basin_mean(basin_data)
        basin_mean = processor.convert_units(basin_mean)
        
        # 日历年聚合
        annual_calendar = processor.aggregate_to_annual(
            basin_mean, method='sum', water_year=False
        )
        
        # 水文年聚合（10月起始）
        annual_hydro = processor.aggregate_to_annual(
            basin_mean, method='sum', water_year=True, water_year_start_month=10
        )
        
        print("\n日历年前5年:")
        print(annual_calendar.head())
        
        print("\n水文年前5年:")
        print(annual_hydro.head())
        
        print(f"\n日历年均值: {annual_calendar.mean():.2f} mm")
        print(f"水文年均值: {annual_hydro.mean():.2f} mm")
        
        return annual_calendar, annual_hydro
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {nc_file}")
        return None, None


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 70)
    print("ISIMIP气候数据处理器使用示例集")
    print("=" * 70)
    
    # 运行各个示例
    example_1_single_variable()
    example_2_complete_pipeline()
    example_3_batch_processing()
    example_4_quality_checks()
    example_5_aridity_classification()
    
    # 可视化示例
    try:
        example_6_visualization()
    except ImportError:
        print("\n⚠️  示例6需要matplotlib库，请运行: pip install matplotlib")
    
    example_7_water_year()


if __name__ == "__main__":
    main()
