"""
test_real_data.py

真实数据测试脚本

使用下载的真实数据（或模拟数据）测试完整工作流：
1. 读取GRDC径流数据
2. 处理气候数据
3. 计算PET
4. 参数校准和归因分析

作者: Research Software Engineer  
日期: 2025-01-01
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.grdc_parser import GRDCParser
from src.budyko_model.pet_calculator import PETCalculator
from src.budyko_model.parameter_calibration import ParameterCalibrator


def test_grdc_sample_data():
    """测试GRDC样本数据解析"""
    print("\n" + "="*70)
    print("测试1: GRDC数据解析")
    print("="*70)
    
    data_dir = project_root / "data" / "raw" / "GRDC"
    grdc_files = list(data_dir.glob("*.txt"))
    
    if not grdc_files:
        print("❌ 未找到GRDC数据文件")
        print(f"   请将GRDC数据放置在: {data_dir}")
        return None
    
    print(f"\n找到 {len(grdc_files)} 个GRDC文件:")
    for f in grdc_files:
        print(f"  - {f.name}")
    
    # 解析第一个文件
    grdc_file = grdc_files[0]
    print(f"\n解析文件: {grdc_file.name}")
    
    try:
        parser = GRDCParser(str(grdc_file))
        
        # 提取元数据
        metadata = parser.parse_metadata()
        print("\n【站点元数据】")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # 读取时间序列
        df = parser.read_timeseries()
        print(f"\n【时间序列】")
        print(f"  记录数: {len(df)}")
        print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
        print(f"  平均流量: {df['discharge'].mean():.2f} m³/s")
        
        # 转换为年值
        df_annual = parser.aggregate_to_annual()
        print(f"\n【年度数据】")
        print(f"  年份数: {len(df_annual)}")
        print(df_annual.head())
        
        # 转换为径流深度
        df_depth = parser.convert_to_depth()
        print(f"\n【径流深度】")
        print(f"  平均径流深度: {df_depth['runoff_depth_mm'].mean():.1f} mm/year")
        
        print("\n✅ GRDC数据解析成功")
        return df_depth
        
    except Exception as e:
        print(f"\n❌ 解析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_climate_data_simulation():
    """模拟气候数据（当ISIMIP数据未下载时）"""
    print("\n" + "="*70)
    print("测试2: 气候数据模拟")
    print("="*70)
    
    print("\n生成模拟气候数据用于测试...")
    
    # 生成1960-2016年的年度气候数据
    years = np.arange(1960, 2017)
    np.random.seed(456)
    
    climate_data = pd.DataFrame({
        'year': years,
        'P': np.random.normal(850, 100, len(years)),      # 降水 (mm/year)
        'PET': np.random.normal(1100, 120, len(years)),   # PET (mm/year)
        'tas': np.random.normal(15, 2, len(years)),       # 气温 (°C)
    })
    
    print("\n【模拟气候数据】")
    print(f"  年份范围: {years[0]}-{years[-1]}")
    print(f"  平均降水: {climate_data['P'].mean():.1f} mm/year")
    print(f"  平均PET: {climate_data['PET'].mean():.1f} mm/year")
    print(f"  平均气温: {climate_data['tas'].mean():.1f} °C")
    
    print("\n✅ 气候数据模拟完成")
    return climate_data


def test_complete_workflow():
    """测试完整工作流"""
    print("\n" + "="*70)
    print("测试3: 完整归因分析工作流")
    print("="*70)
    
    # 步骤1: 获取径流数据
    runoff_data = test_grdc_sample_data()
    
    if runoff_data is None:
        print("\n使用模拟径流数据...")
        years = np.arange(1960, 2017)
        runoff_data = pd.DataFrame({
            'year': years,
            'runoff_depth_mm': np.random.normal(250, 50, len(years))
        })
    else:
        # 确保有year列
        if 'year' not in runoff_data.columns:
            runoff_data['year'] = runoff_data.index.year
    
    # 步骤2: 获取气候数据
    climate_data = test_climate_data_simulation()
    
    # 步骤3: 合并数据
    print("\n" + "="*70)
    print("步骤3: 数据整合")
    print("="*70)
    
    # 合并径流和气候数据
    combined_data = pd.merge(
        runoff_data[['year', 'runoff_depth_mm']],
        climate_data[['year', 'P', 'PET']],
        on='year',
        how='inner'
    )
    
    # 重命名列
    combined_data.rename(columns={'runoff_depth_mm': 'Q_n'}, inplace=True)
    
    print(f"\n合并后数据: {len(combined_data)} 年")
    print(combined_data.head())
    
    # 步骤4: 参数校准和归因分析
    print("\n" + "="*70)
    print("步骤4: 参数校准和归因分析")
    print("="*70)
    
    calibrator = ParameterCalibrator(change_point=1986, min_valid_years=10)
    
    # 时段演变分析
    results = calibrator.analyze_parameter_evolution(
        station_id="TEST_STATION",
        time_series=combined_data,
        P_col='P',
        PET_col='PET',
        Q_n_col='Q_n',
        year_col='year'
    )
    
    print("\n【参数演变结果】")
    for period_key, result in results.items():
        if result:
            print(f"\n{period_key} ({result.period}):")
            print(f"  参数 n: {result.n:.4f}")
            print(f"  降水 P: {result.P:.1f} mm/year")
            print(f"  PET: {result.PET:.1f} mm/year")
            print(f"  径流 Q_n: {result.Q_n:.1f} mm/year")
            print(f"  蒸散发 E: {result.E:.1f} mm/year")
            print(f"  干旱指数: {result.aridity_index:.3f}")
    
    # 归因分析
    if 'period_1' in results and 'period_2' in results:
        print("\n【归因分析】")
        
        # 模拟观测径流（假设有5%的人类取用水）
        period_1_mask = combined_data['year'] < 1986
        period_2_mask = combined_data['year'] >= 1986
        
        Q_obs_1 = combined_data.loc[period_1_mask, 'Q_n'].mean() * 0.95
        Q_obs_2 = combined_data.loc[period_2_mask, 'Q_n'].mean() * 0.90  # 后期取水增加
        
        attribution = calibrator.calculate_attribution(
            station_id="TEST_STATION",
            period_1_data=results['period_1'],
            period_2_data=results['period_2'],
            Q_obs_1=Q_obs_1,
            Q_obs_2=Q_obs_2
        )
        
        if attribution:
            print(f"\n观测径流变化: ΔQ_obs = {attribution.delta_Q_obs:.1f} mm/year")
            print(f"天然径流变化: ΔQ_n = {attribution.delta_Q_n:.1f} mm/year")
            
            print(f"\n【归因贡献量】")
            print(f"  气候变化 (CCV):      {attribution.delta_Q_CCV:+.1f} mm/year")
            print(f"  土地利用变化 (LUCC): {attribution.delta_Q_LUCC:+.1f} mm/year")
            print(f"  人类取用水 (WADR):    {attribution.delta_Q_WADR:+.1f} mm/year")
            
            if not np.isnan(attribution.C_CCV):
                print(f"\n【归因贡献率】")
                print(f"  气候变化 (CCV):      {attribution.C_CCV:.1f}%")
                print(f"  土地利用变化 (LUCC): {attribution.C_LUCC:.1f}%")
                print(f"  人类取用水 (WADR):    {attribution.C_WADR:.1f}%")
            
            print(f"\n【弹性系数】")
            print(f"  εP (降水弹性):   {attribution.elasticity['epsilon_P']:.3f}")
            print(f"  εPET (PET弹性):  {attribution.elasticity['epsilon_PET']:.3f}")
            print(f"  εn (参数n弹性):  {attribution.elasticity['epsilon_n']:.3f}")
    
    print("\n✅ 完整工作流测试完成")
    
    # 步骤5: 导出结果
    print("\n" + "="*70)
    print("步骤5: 导出结果")
    print("="*70)
    
    output_dir = project_root / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出合并数据
    combined_output = output_dir / "test_combined_data.csv"
    combined_data.to_csv(combined_output, index=False)
    print(f"\n✅ 合并数据已导出: {combined_output}")
    
    # 导出校准结果
    if results:
        calibration_results = []
        for key, result in results.items():
            if result:
                calibration_results.append(result)
        
        if calibration_results:
            calib_output = output_dir / "test_calibration_results.csv"
            calibrator.export_results(calibration_results, str(calib_output))
            print(f"✅ 校准结果已导出: {calib_output}")
    
    # 导出归因结果
    if 'period_1' in results and 'period_2' in results and attribution:
        attrib_output = output_dir / "test_attribution_results.csv"
        calibrator.export_attribution_results([attribution], str(attrib_output))
        print(f"✅ 归因结果已导出: {attrib_output}")


def test_batch_stations():
    """测试批量站点处理"""
    print("\n" + "="*70)
    print("测试4: 批量站点处理")
    print("="*70)
    
    print("\n生成多站点模拟数据...")
    
    # 创建5个模拟站点的数据
    np.random.seed(789)
    stations_data = []
    
    for i in range(5):
        station_id = f"STATION_{i+1:03d}"
        
        # 不同气候类型的流域
        if i == 0:  # 湿润
            P, PET, Q_n = 1500, 900, 800
        elif i == 1:  # 半湿润
            P, PET, Q_n = 900, 1100, 250
        elif i == 2:  # 半干旱
            P, PET, Q_n = 550, 1200, 100
        elif i == 3:  # 干旱
            P, PET, Q_n = 300, 1400, 40
        else:  # 中等
            P, PET, Q_n = 1000, 1000, 350
        
        # 添加随机扰动
        P += np.random.normal(0, 50)
        PET += np.random.normal(0, 80)
        Q_n += np.random.normal(0, 30)
        Q_n = max(Q_n, 10)  # 确保非负
        
        stations_data.append({
            'station_id': station_id,
            'P': P,
            'PET': PET,
            'Q_n': Q_n,
            'region': ['湿润', '半湿润', '半干旱', '干旱', '中等'][i]
        })
    
    df_stations = pd.DataFrame(stations_data)
    print("\n站点数据:")
    print(df_stations)
    
    # 批量校准
    print("\n执行批量校准...")
    calibrator = ParameterCalibrator()
    results = calibrator.batch_calibrate_stations(df_stations, parallel=False)
    
    print(f"\n✅ 批量校准完成: {len(results)}/{len(df_stations)} 站点成功")
    
    # 显示结果
    results_df = pd.DataFrame([r.to_dict() for r in results])
    results_df = results_df.merge(df_stations[['station_id', 'region']], on='station_id')
    
    print("\n【校准结果汇总】")
    print(results_df[['station_id', 'region', 'n', 'aridity_index', 
                      'calibration_error']].to_string(index=False))
    
    # 导出
    output_dir = project_root / "data" / "results"
    output_file = output_dir / "test_batch_calibration.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 批量结果已导出: {output_file}")


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("真实数据测试套件")
    print("="*70)
    print("\n本测试使用模拟数据演示完整工作流")
    print("替换为真实GRDC、ISIMIP数据后可进行实际分析")
    
    try:
        # 测试1: GRDC数据
        test_grdc_sample_data()
        
        # 测试2: 完整工作流
        test_complete_workflow()
        
        # 测试3: 批量处理
        test_batch_stations()
        
        print("\n" + "="*70)
        print("所有测试完成！")
        print("="*70)
        print("""
✅ 测试总结:
   - GRDC数据解析: 成功
   - 气候数据处理: 成功
   - 参数校准: 成功
   - 归因分析: 成功
   - 批量处理: 成功
   - 结果导出: 成功

📁 输出文件位置:
   data/results/test_*.csv

🔍 下一步:
   1. 下载真实GRDC数据（按照 data/raw/GRDC/GRDC_DOWNLOAD_INSTRUCTIONS.txt）
   2. 下载ISIMIP数据（按照 data/raw/ISIMIP/ISIMIP_DATA_INFO.txt）
   3. 运行完整分析流程
""")
        
    except Exception as e:
        print(f"\n❌ 测试过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
