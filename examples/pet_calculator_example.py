"""
PET计算器使用示例
===============

演示如何使用PETCalculator计算潜在蒸散发(PET)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.budyko_model.pet_calculator import (
    PETCalculator,
    convert_rsds_to_mj,
    estimate_missing_tmax_tmin,
    validate_pet_reasonableness,
    calculate_aridity_classification
)


def example_1_single_point():
    """示例1: 单点PET计算"""
    print("=" * 70)
    print("示例1: 单点日尺度PET计算")
    print("=" * 70)
    
    # 初始化计算器
    # 假设站点位于长江中游（武汉附近）
    calculator = PETCalculator(
        latitude=30.5,  # 纬度
        elevation=50,   # 海拔50米
        method='pm'     # FAO-56 Penman-Monteith方法
    )
    
    print(f"\n站点信息:")
    print(f"  纬度: {calculator.latitude}°")
    print(f"  海拔: {calculator.elevation} m")
    print(f"  方法: {calculator.method}")
    
    # 输入气象数据（夏季典型日）
    tmean = 28.0  # 平均气温 (°C)
    tmax = 33.0   # 最高气温 (°C)
    tmin = 23.0   # 最低气温 (°C)
    rs = 22.0     # 太阳辐射 (MJ/m²/day)
    rh = 70.0     # 相对湿度 (%)
    uz = 2.5      # 风速 (m/s)
    
    print(f"\n输入气象数据:")
    print(f"  平均气温: {tmean} °C")
    print(f"  最高气温: {tmax} °C")
    print(f"  最低气温: {tmin} °C")
    print(f"  太阳辐射: {rs} MJ/m²/day")
    print(f"  相对湿度: {rh} %")
    print(f"  风速: {uz} m/s")
    
    # 计算PET
    pet = calculator.calculate_fao56(
        tmean=tmean,
        tmax=tmax,
        tmin=tmin,
        rs=rs,
        rh=rh,
        uz=uz
    )
    
    print(f"\n计算结果:")
    print(f"  PET: {pet:.2f} mm/day")
    print(f"  年PET(×365): {pet * 365:.2f} mm/year")
    
    return pet


def example_2_time_series():
    """示例2: 时间序列PET计算"""
    print("\n" + "=" * 70)
    print("示例2: 年度时间序列PET计算")
    print("=" * 70)
    
    calculator = PETCalculator(latitude=30.5, elevation=50)
    
    # 生成2020年全年的模拟气象数据
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    n = len(dates)
    
    # 模拟季节变化
    day_of_year = np.arange(n)
    tmean = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    tmax = tmean + 5 + 2 * np.random.randn(n)
    tmin = tmean - 5 + 2 * np.random.randn(n)
    rs = 15 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 3 * np.random.randn(n)
    rs = np.maximum(rs, 5)  # 确保非负
    
    # 转换为Series
    tmean_series = pd.Series(tmean, index=dates)
    tmax_series = pd.Series(tmax, index=dates)
    tmin_series = pd.Series(tmin, index=dates)
    rs_series = pd.Series(rs, index=dates)
    
    print(f"\n生成{len(dates)}天的模拟气象数据")
    print(f"  气温范围: {tmean_series.min():.1f} - {tmean_series.max():.1f} °C")
    print(f"  辐射范围: {rs_series.min():.1f} - {rs_series.max():.1f} MJ/m²/day")
    
    # 计算日PET
    pet_daily = calculator.calculate_fao56(
        tmean=tmean_series,
        tmax=tmax_series,
        tmin=tmin_series,
        rs=rs_series
    )
    
    print(f"\n日PET统计:")
    print(f"  最小值: {pet_daily.min():.2f} mm/day")
    print(f"  最大值: {pet_daily.max():.2f} mm/day")
    print(f"  平均值: {pet_daily.mean():.2f} mm/day")
    
    # 聚合为年PET
    pet_annual = calculator.aggregate_to_annual(pet_daily, method='sum')
    
    print(f"\n年PET:")
    print(f"  2020年总量: {pet_annual.iloc[0]:.2f} mm/year")
    
    return pet_daily, pet_annual


def example_3_hargreaves_method():
    """示例3: Hargreaves简化方法（缺少辐射、湿度数据时使用）"""
    print("\n" + "=" * 70)
    print("示例3: Hargreaves简化方法")
    print("=" * 70)
    
    calculator = PETCalculator(latitude=30.5)
    
    print("\n当缺少辐射、湿度、风速数据时，可使用Hargreaves方法")
    print("该方法仅需要气温数据")
    
    # 仅使用气温数据
    tmean = 25.0
    tmax = 30.0
    tmin = 20.0
    
    pet_hargreaves = calculator.calculate_hargreaves(
        tmean=tmean,
        tmax=tmax,
        tmin=tmin
    )
    
    # 对比FAO-56方法
    pet_fao56 = calculator.calculate_fao56(
        tmean=tmean,
        tmax=tmax,
        tmin=tmin,
        rs=20.0  # 假设中等辐射
    )
    
    print(f"\n气温数据:")
    print(f"  tmean: {tmean} °C, tmax: {tmax} °C, tmin: {tmin} °C")
    
    print(f"\n计算结果对比:")
    print(f"  Hargreaves方法: {pet_hargreaves:.2f} mm/day")
    print(f"  FAO-56方法:     {pet_fao56:.2f} mm/day")
    print(f"  差异:           {abs(pet_hargreaves - pet_fao56):.2f} mm/day")


def example_4_isimip_integration():
    """示例4: 与ISIMIP气候数据集成"""
    print("\n" + "=" * 70)
    print("示例4: 与ISIMIP气候数据集成")
    print("=" * 70)
    
    print("\n模拟从ISIMIP处理后的气候数据计算PET")
    
    calculator = PETCalculator(latitude=30.5, elevation=50)
    
    # 模拟climate_processor的输出
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    climate_data = {
        'tas': pd.Series(20 + 5 * np.sin(np.linspace(0, 2*np.pi, 365)), index=dates),
        'tasmax': pd.Series(25 + 5 * np.sin(np.linspace(0, 2*np.pi, 365)), index=dates),
        'tasmin': pd.Series(15 + 5 * np.sin(np.linspace(0, 2*np.pi, 365)), index=dates),
        'rsds': pd.Series(18 + 6 * np.sin(np.linspace(0, 2*np.pi, 365)), index=dates),
        'hurs': pd.Series(65.0, index=dates),
        'sfcWind': pd.Series(2.0, index=dates)
    }
    
    print(f"\n气候数据变量: {list(climate_data.keys())}")
    
    # 计算PET
    pet = calculator.calculate_from_climate_data(climate_data)
    
    # 聚合为年值
    pet_annual = calculator.aggregate_to_annual(pet, method='sum')
    
    print(f"\nPET计算结果:")
    print(f"  日均PET: {pet.mean():.2f} mm/day")
    print(f"  年PET:   {pet_annual.iloc[0]:.2f} mm/year")
    
    return pet


def example_5_unit_conversion():
    """示例5: ISIMIP单位转换"""
    print("\n" + "=" * 70)
    print("示例5: ISIMIP辐射单位转换")
    print("=" * 70)
    
    print("\nISIMIP原始辐射单位: W/m²")
    print("FAO-56所需单位: MJ/m²/day")
    
    # ISIMIP典型辐射值
    rsds_wm2 = np.array([150, 200, 250, 300])
    
    # 转换为MJ/m²/day
    rs_mj = convert_rsds_to_mj(rsds_wm2)
    
    print(f"\n转换示例:")
    for w, m in zip(rsds_wm2, rs_mj):
        print(f"  {w:>3.0f} W/m² → {m:>6.2f} MJ/m²/day")
    
    print(f"\n转换公式: MJ/m²/day = W/m² × 0.0864")


def example_6_data_quality_check():
    """示例6: PET数据质量检查"""
    print("\n" + "=" * 70)
    print("示例6: PET数据质量检查")
    print("=" * 70)
    
    test_cases = [
        {'pet': 1500, 'p': 1000, 'desc': '正常流域'},
        {'pet': 2000, 'p': 500, 'desc': '半干旱流域'},
        {'pet': 3000, 'p': 200, 'desc': '干旱流域'},
        {'pet': -100, 'p': 1000, 'desc': '异常：负PET'},
        {'pet': 6000, 'p': 1000, 'desc': '异常：极端高PET'},
        {'pet': 8000, 'p': 100, 'desc': '异常：极端干旱指数'},
    ]
    
    print(f"\n{'描述':<20} {'PET(mm)':<10} {'P(mm)':<8} {'状态':<6} {'说明'}")
    print("-" * 70)
    
    for case in test_cases:
        is_valid, msg = validate_pet_reasonableness(case['pet'], case['p'])
        status = "✓" if is_valid else "✗"
        print(f"{case['desc']:<20} {case['pet']:<10} {case['p']:<8} {status:<6} {msg}")


def example_7_aridity_classification():
    """示例7: 干旱指数与气候分类"""
    print("\n" + "=" * 70)
    print("示例7: 基于干旱指数的气候分类")
    print("=" * 70)
    
    basins = [
        {'name': '长江流域', 'p': 1100, 'pet': 900},
        {'name': '黄河流域', 'p': 450, 'pet': 1200},
        {'name': '塔里木河流域', 'p': 150, 'pet': 1800},
        {'name': '珠江流域', 'p': 1800, 'pet': 1100},
        {'name': '海河流域', 'p': 500, 'pet': 1300},
    ]
    
    print(f"\n{'流域':<15} {'P(mm)':<8} {'PET(mm)':<10} {'干旱指数':<12} {'气候类型'}")
    print("-" * 70)
    
    for basin in basins:
        ai, climate = calculate_aridity_classification(basin['pet'], basin['p'])
        print(f"{basin['name']:<15} {basin['p']:<8} {basin['pet']:<10} {ai:<12.2f} {climate}")


def example_8_visualization():
    """示例8: PET数据可视化"""
    print("\n" + "=" * 70)
    print("示例8: PET季节变化可视化")
    print("=" * 70)
    
    calculator = PETCalculator(latitude=30.5)
    
    # 生成全年数据
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    day_of_year = np.arange(len(dates))
    
    # 季节性气温和辐射
    tmean = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    tmax = tmean + 5
    tmin = tmean - 5
    rs = 15 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    tmean_series = pd.Series(tmean, index=dates)
    tmax_series = pd.Series(tmax, index=dates)
    tmin_series = pd.Series(tmin, index=dates)
    rs_series = pd.Series(rs, index=dates)
    
    # 计算PET
    pet = calculator.calculate_fao56(tmean_series, tmax_series, tmin_series, rs_series)
    
    print("\n绘制PET季节变化图...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 子图1: 日PET时间序列
    axes[0].plot(pet.index, pet.values, color='steelblue', linewidth=1.5)
    axes[0].set_ylabel('日PET (mm/day)', fontsize=12)
    axes[0].set_title('长江流域2020年日PET时间序列（示例）', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=pet.mean(), color='red', linestyle='--', 
                   label=f'平均值: {pet.mean():.2f} mm/day')
    axes[0].legend()
    
    # 子图2: 月PET柱状图
    pet_monthly = pet.resample('ME').sum()
    pet_monthly.index = pet_monthly.index.month
    
    axes[1].bar(pet_monthly.index, pet_monthly.values, 
               color='darkorange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('月份', fontsize=12)
    axes[1].set_ylabel('月PET (mm/month)', fontsize=12)
    axes[1].set_title('月度PET分布', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = "figures/pet_seasonal_example.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {output_file}")
    
    plt.show()


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 70)
    print("PET计算器使用示例集")
    print("=" * 70)
    
    # 运行各示例
    example_1_single_point()
    example_2_time_series()
    example_3_hargreaves_method()
    example_4_isimip_integration()
    example_5_unit_conversion()
    example_6_data_quality_check()
    example_7_aridity_classification()
    
    # 可视化示例
    try:
        example_8_visualization()
    except ImportError:
        print("\n⚠️  示例8需要matplotlib库")


if __name__ == "__main__":
    main()
