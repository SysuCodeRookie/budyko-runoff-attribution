"""
GRDC解析器使用示例
================

演示如何使用GRDCParser处理观测径流数据
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.grdc_parser import GRDCParser, validate_runoff_ratio


def example_1_basic_usage():
    """示例1: 基本用法 - 单个站点"""
    print("=" * 60)
    print("示例1: 解析单个GRDC站点")
    print("=" * 60)
    
    # 替换为您的实际文件路径
    grdc_file = "data/raw/GRDC/6335020_Q_Day.Cmd.txt"
    
    try:
        # 初始化解析器
        parser = GRDCParser(grdc_file)
        
        # 1. 提取元数据
        print("\n1. 提取站点元数据...")
        metadata = parser.parse_metadata()
        print(f"   站点编号: {metadata['grdc_no']}")
        print(f"   站点名称: {metadata['station']}")
        print(f"   河流: {metadata['river']}")
        print(f"   国家: {metadata['country']}")
        print(f"   坐标: ({metadata['latitude']:.2f}°, {metadata['longitude']:.2f}°)")
        print(f"   集水区面积: {metadata['area_km2']:,.0f} km²")
        print(f"   数据时段: {metadata['time_series_start']} 至 {metadata['time_series_end']}")
        
        # 2. 读取时间序列
        print("\n2. 读取时间序列数据...")
        df_daily = parser.read_timeseries()
        print(f"   数据点数: {len(df_daily)}")
        print(f"   有效数据: {df_daily['discharge_m3s'].notna().sum()}")
        print(f"   缺测率: {df_daily['discharge_m3s'].isna().sum() / len(df_daily) * 100:.1f}%")
        
        # 3. 单位转换
        print("\n3. 转换为径流深度...")
        df_depth = parser.convert_to_depth()
        print(f"   日均径流深度: {df_depth['runoff_mm_year'].mean():.2f} mm/year")
        
        # 4. 年度聚合
        print("\n4. 聚合为年值...")
        df_annual = parser.aggregate_to_annual()
        print(f"   年份数: {len(df_annual)}")
        print(f"   年均流量: {df_annual['discharge_m3s'].mean():.0f} m³/s")
        
        # 5. 质量过滤
        print("\n5. 应用质量过滤...")
        df_filtered = parser.quality_filter(max_missing_pct=15)
        print(f"   过滤后年份数: {len(df_filtered)}")
        
        return df_filtered
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {grdc_file}")
        print("   请将文件路径替换为您的实际GRDC数据文件")
        return None


def example_2_batch_processing():
    """示例2: 批量处理多个站点"""
    print("\n" + "=" * 60)
    print("示例2: 批量加载多个GRDC站点")
    print("=" * 60)
    
    grdc_dir = "data/raw/GRDC/"
    
    try:
        # 批量加载
        stations = GRDCParser.load_multiple_stations(grdc_dir)
        
        print(f"\n成功加载 {len(stations)} 个站点\n")
        
        # 遍历所有站点
        results = []
        for grdc_no, parser in stations.items():
            metadata = parser.parse_metadata()
            annual = parser.aggregate_to_annual()
            
            results.append({
                'grdc_no': grdc_no,
                'station': metadata['station'],
                'area_km2': metadata['area_km2'],
                'mean_discharge': annual['discharge_m3s'].mean(),
                'years': len(annual)
            })
            
            print(f"站点 {grdc_no:7d} | {metadata['station']:20s} | "
                  f"面积: {metadata['area_km2']:>10,.0f} km² | "
                  f"年均流量: {annual['discharge_m3s'].mean():>8,.0f} m³/s")
        
        # 转为DataFrame
        df_summary = pd.DataFrame(results)
        return df_summary
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        return None


def example_3_water_year():
    """示例3: 水文年处理"""
    print("\n" + "=" * 60)
    print("示例3: 使用水文年进行聚合")
    print("=" * 60)
    
    grdc_file = "data/raw/GRDC/6335020_Q_Day.Cmd.txt"
    
    try:
        parser = GRDCParser(grdc_file)
        parser.parse_metadata()
        parser.read_timeseries()
        
        # 日历年聚合
        annual_calendar = parser.aggregate_to_annual(water_year=False)
        print("\n日历年聚合（1月-12月）:")
        print(annual_calendar.head())
        
        # 水文年聚合（10月-9月）
        annual_water = parser.aggregate_to_annual(
            water_year=True, 
            water_year_start_month=10
        )
        print("\n水文年聚合（10月-9月）:")
        print(annual_water.head())
        
        return annual_calendar, annual_water
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {grdc_file}")
        return None, None


def example_4_runoff_ratio_check():
    """示例4: 径流系数验证"""
    print("\n" + "=" * 60)
    print("示例4: 径流系数合理性检验")
    print("=" * 60)
    
    # 模拟数据
    test_cases = [
        (400, 800, "正常情况"),
        (100, 1000, "极端干旱"),
        (950, 1000, "极端湿润"),
        (1100, 1000, "异常: 径流>降水"),
        (-50, 800, "异常: 负径流"),
    ]
    
    print("\n径流深度 | 降水量 | 径流系数 | 检验结果")
    print("-" * 60)
    
    for runoff, precip, description in test_cases:
        is_valid, message = validate_runoff_ratio(runoff, precip)
        status = "✓" if is_valid else "✗"
        print(f"{runoff:>8} mm | {precip:>6} mm | {runoff/precip:>8.3f} | {status} {message}")


def example_5_visualization():
    """示例5: 数据可视化"""
    print("\n" + "=" * 60)
    print("示例5: 绘制径流时间序列图")
    print("=" * 60)
    
    grdc_file = "data/raw/GRDC/6335020_Q_Day.Cmd.txt"
    
    try:
        parser = GRDCParser(grdc_file)
        metadata = parser.parse_metadata()
        df_annual = parser.aggregate_to_annual()
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 子图1: 年际变化
        axes[0].plot(df_annual['year'], df_annual['discharge_m3s'], 
                    marker='o', linewidth=2, markersize=4)
        axes[0].set_xlabel('年份', fontsize=12)
        axes[0].set_ylabel('年均流量 (m³/s)', fontsize=12)
        axes[0].set_title(f"站点 {metadata['station']} - 径流年际变化", 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 子图2: 数据完整性
        axes[1].bar(df_annual['year'], df_annual['data_count'], 
                   color='steelblue', alpha=0.7)
        axes[1].axhline(y=365, color='red', linestyle='--', 
                       label='完整年份 (365天)')
        axes[1].set_xlabel('年份', fontsize=12)
        axes[1].set_ylabel('有效观测天数', fontsize=12)
        axes[1].set_title('数据完整性检查', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = "figures/grdc_timeseries_example.png"
        Path("figures").mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {output_file}")
        
        plt.show()
        
    except FileNotFoundError:
        print(f"\n⚠️  文件不存在: {grdc_file}")
    except Exception as e:
        print(f"\n⚠️  可视化失败: {e}")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("GRDC解析器使用示例集")
    print("=" * 60)
    
    # 运行各个示例
    example_1_basic_usage()
    example_2_batch_processing()
    example_3_water_year()
    example_4_runoff_ratio_check()
    
    # 可视化示例需要matplotlib
    try:
        example_5_visualization()
    except ImportError:
        print("\n⚠️  示例5需要matplotlib库，请运行: pip install matplotlib")


if __name__ == "__main__":
    main()
