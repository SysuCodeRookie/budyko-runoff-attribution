"""
parameter_calibration_example.py

模块5使用示例：批量站点参数校准和归因分析

本示例展示了如何使用ParameterCalibrator类进行：
1. 单站点参数校准
2. 批量多站点处理
3. 时间演变分析（参数n的变化）
4. 完整的归因分解（CCV、LUCC、WADR）
5. Bootstrap不确定性估计
6. 集成归因统计

作者: Research Software Engineer
日期: 2025-01-01
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.budyko_model.parameter_calibration import (
    ParameterCalibrator,
    validate_time_series_quality,
    calculate_ensemble_attribution
)


def example_1_single_station_calibration():
    """
    示例1：单站点参数校准
    
    展示如何对单个流域进行Budyko参数n的反演计算。
    """
    print("\n" + "="*70)
    print("示例1：单站点参数校准")
    print("="*70)
    
    calibrator = ParameterCalibrator()
    
    # 长江上游某流域的假想数据
    station_id = "Yangtze_Upper_001"
    P = 1050.0  # mm/year
    PET = 950.0  # mm/year
    Q_n = 400.0  # mm/year
    
    print(f"\n站点：{station_id}")
    print(f"多年平均降水量 (P): {P:.1f} mm/year")
    print(f"多年平均潜在蒸散发 (PET): {PET:.1f} mm/year")
    print(f"多年平均天然径流 (Q_n): {Q_n:.1f} mm/year")
    
    result = calibrator.calibrate_single_station(
        station_id=station_id,
        P=P,
        PET=PET,
        Q_n=Q_n,
        period="1960-2016"
    )
    
    if result:
        print(f"\n【校准结果】")
        print(f"流域景观参数 (n): {result.n:.4f}")
        print(f"干旱指数 (PET/P): {result.aridity_index:.3f}")
        print(f"计算的实际蒸散发 (E): {result.E:.1f} mm/year")
        print(f"水量平衡检查 (P - E - Q_n): {P - result.E - Q_n:.4f} mm/year")
        print(f"校准误差: {result.calibration_error:.4f}%")
        
        print(f"\n【物理解释】")
        if result.aridity_index < 1.0:
            print("该流域属于湿润气候（能量限制型）")
        else:
            print("该流域属于干旱/半干旱气候（水分限制型）")
        
        print(f"参数n={result.n:.2f}表明流域具有{'较强' if result.n > 2.5 else '中等' if result.n > 1.5 else '较弱'}的"
              "蒸散发能力")
    else:
        print("校准失败：可能违反了水量平衡约束")


def example_2_batch_calibration():
    """
    示例2：批量多站点校准
    
    展示如何同时处理多个流域，适用于区域尺度研究。
    """
    print("\n" + "="*70)
    print("示例2：批量多站点校准")
    print("="*70)
    
    calibrator = ParameterCalibrator()
    
    # 创建多个站点的数据（模拟中国不同气候区的流域）
    stations_data = pd.DataFrame({
        'station_id': [
            'Northeast_Humid',
            'Northwest_Arid',
            'Southeast_Humid',
            'Southwest_Mountainous',
            'NorthChina_SemiArid'
        ],
        'P': [600, 250, 1600, 1100, 550],
        'PET': [650, 1400, 1100, 850, 1000],
        'Q_n': [180, 30, 800, 450, 120],
        'region': ['东北', '西北', '东南', '西南', '华北']
    })
    
    print(f"\n待校准站点数: {len(stations_data)}")
    print("\n站点基本信息:")
    print(stations_data[['station_id', 'region', 'P', 'PET', 'Q_n']].to_string(index=False))
    
    # 批量校准
    results = calibrator.batch_calibrate_stations(
        stations_data,
        parallel=False  # 在示例中使用顺序处理
    )
    
    print(f"\n【校准完成】成功校准 {len(results)}/{len(stations_data)} 个站点")
    
    # 整理结果
    results_df = pd.DataFrame([r.to_dict() for r in results])
    results_df = results_df.merge(
        stations_data[['station_id', 'region']], 
        on='station_id'
    )
    
    print("\n校准结果汇总:")
    print(results_df[['station_id', 'region', 'n', 'aridity_index', 
                      'calibration_error']].to_string(index=False))
    
    # 统计分析
    print(f"\n【统计摘要】")
    print(f"参数n范围: {results_df['n'].min():.2f} - {results_df['n'].max():.2f}")
    print(f"平均参数n: {results_df['n'].mean():.2f} ± {results_df['n'].std():.2f}")
    print(f"干旱指数范围: {results_df['aridity_index'].min():.2f} - "
          f"{results_df['aridity_index'].max():.2f}")


def example_3_parameter_evolution():
    """
    示例3：参数演变分析
    
    分析流域参数n在不同时段的变化，揭示LUCC信号。
    """
    print("\n" + "="*70)
    print("示例3：参数演变分析（检测土地利用变化信号）")
    print("="*70)
    
    calibrator = ParameterCalibrator(change_point=1986)
    
    # 创建1960-2016年的时间序列
    # 模拟一个经历了大规模植树造林的流域
    np.random.seed(123)
    years = np.arange(1960, 2017)
    
    # Period 1 (1960-1985): 较低的植被覆盖
    # Period 2 (1986-2016): 植被恢复，蒸散发增强
    P_base = 850.0
    PET_base = 1150.0
    
    time_series = pd.DataFrame({
        'year': years,
        'P': np.random.normal(P_base, 60, len(years)),
        'PET': np.random.normal(PET_base, 70, len(years)),
        'Q_n': np.concatenate([
            np.random.normal(280, 25, 26),  # 1960-1985: 较高径流
            np.random.normal(220, 25, 31)   # 1986-2016: 径流减少（更多ET）
        ])
    })
    
    station_id = "Loess_Plateau_Reforestation"
    print(f"\n站点：{station_id}")
    print(f"数据时段: {years[0]}-{years[-1]} ({len(years)} 年)")
    print(f"突变点: {calibrator.change_point}")
    
    results = calibrator.analyze_parameter_evolution(
        station_id=station_id,
        time_series=time_series
    )
    
    print(f"\n【参数演变结果】")
    for period_key, result in results.items():
        print(f"\n{period_key.upper()} ({result.period}):")
        print(f"  参数 n: {result.n:.4f}")
        print(f"  降水 P: {result.P:.1f} mm/year")
        print(f"  PET: {result.PET:.1f} mm/year")
        print(f"  径流 Q_n: {result.Q_n:.1f} mm/year")
        print(f"  蒸散发 E: {result.E:.1f} mm/year")
        print(f"  径流系数 (Q/P): {result.Q_n/result.P:.3f}")
    
    # 分析参数变化
    if 'period_1' in results and 'period_2' in results:
        delta_n = results['period_2'].n - results['period_1'].n
        delta_Q = results['period_2'].Q_n - results['period_1'].Q_n
        
        print(f"\n【变化检测】")
        print(f"参数n变化: {results['period_1'].n:.3f} → {results['period_2'].n:.3f} "
              f"(Δn = {delta_n:+.3f})")
        print(f"径流变化: {results['period_1'].Q_n:.1f} → {results['period_2'].Q_n:.1f} "
              f"(ΔQ = {delta_Q:+.1f} mm/year)")
        
        print(f"\n【物理解释】")
        if delta_n > 0.1:
            print(f"参数n显著增加，表明流域蒸散发能力增强。")
            print(f"可能原因：植被恢复、造林工程、农业灌溉增加等。")
            if delta_Q < 0:
                print(f"径流减少与n增加一致，证实了LUCC对水文过程的影响。")
        else:
            print(f"参数n变化不显著，流域下垫面特征相对稳定。")


def example_4_complete_attribution():
    """
    示例4：完整归因分析
    
    实现main.tex中的归因框架，分解CCV、LUCC、WADR的贡献。
    """
    print("\n" + "="*70)
    print("示例4：完整归因分析（径流变化归因分解）")
    print("="*70)
    
    calibrator = ParameterCalibrator(change_point=1986)
    
    # 模拟黄河流域某支流的数据
    station_id = "Yellow_River_Tributary"
    
    # 创建两个时段的数据
    # Period 1: 1960-1985 (基准期)
    period_1_data = {
        'P': 420.0,
        'PET': 1050.0,
        'Q_n': 85.0,  # 天然径流
        'Q_obs': 85.0  # 观测径流（假设无人类取水）
    }
    
    # Period 2: 1986-2016 (影响期)
    # 气候变化：降水减少，PET增加
    # 土地利用变化：梯田建设，n增加
    # 人类取用水：农业灌溉和城市用水增加
    period_2_data = {
        'P': 380.0,  # 降水减少 -40 mm
        'PET': 1120.0,  # PET增加 +70 mm
        'Q_n': 52.0,  # 天然径流减少
        'Q_obs': 38.0  # 观测径流进一步减少（人类取水）
    }
    
    print(f"\n站点：{station_id}")
    print(f"\nPeriod 1 (1960-1985 基准期):")
    print(f"  P = {period_1_data['P']:.0f} mm/year")
    print(f"  PET = {period_1_data['PET']:.0f} mm/year")
    print(f"  Q_n = {period_1_data['Q_n']:.0f} mm/year (天然径流)")
    print(f"  Q_obs = {period_1_data['Q_obs']:.0f} mm/year (观测径流)")
    
    print(f"\nPeriod 2 (1986-2016 影响期):")
    print(f"  P = {period_2_data['P']:.0f} mm/year (变化: {period_2_data['P'] - period_1_data['P']:+.0f})")
    print(f"  PET = {period_2_data['PET']:.0f} mm/year (变化: {period_2_data['PET'] - period_1_data['PET']:+.0f})")
    print(f"  Q_n = {period_2_data['Q_n']:.0f} mm/year (变化: {period_2_data['Q_n'] - period_1_data['Q_n']:+.0f})")
    print(f"  Q_obs = {period_2_data['Q_obs']:.0f} mm/year (变化: {period_2_data['Q_obs'] - period_1_data['Q_obs']:+.0f})")
    
    # 校准两个时段的参数n
    result_1 = calibrator.calibrate_single_station(
        station_id, period_1_data['P'], period_1_data['PET'], 
        period_1_data['Q_n'], period="1960-1985"
    )
    
    result_2 = calibrator.calibrate_single_station(
        station_id, period_2_data['P'], period_2_data['PET'],
        period_2_data['Q_n'], period="1986-2016"
    )
    
    if result_1 and result_2:
        print(f"\n【参数校准结果】")
        print(f"Period 1: n = {result_1.n:.4f}")
        print(f"Period 2: n = {result_2.n:.4f}")
        print(f"变化: Δn = {result_2.n - result_1.n:+.4f}")
        
        # 计算归因
        attribution = calibrator.calculate_attribution(
            station_id=station_id,
            period_1_data=result_1,
            period_2_data=result_2,
            Q_obs_1=period_1_data['Q_obs'],
            Q_obs_2=period_2_data['Q_obs']
        )
        
        if attribution:
            print(f"\n【归因分解结果】")
            print(f"\n观测径流变化: ΔQ_obs = {attribution.delta_Q_obs:.1f} mm/year")
            print(f"天然径流变化: ΔQ_n = {attribution.delta_Q_n:.1f} mm/year")
            
            print(f"\n各驱动因子贡献量:")
            print(f"  气候变化 (CCV):      ΔQ_CCV = {attribution.delta_Q_CCV:.1f} mm/year")
            print(f"  土地利用变化 (LUCC): ΔQ_LUCC = {attribution.delta_Q_LUCC:.1f} mm/year")
            print(f"  人类取用水 (WADR):    ΔQ_WADR = {attribution.delta_Q_WADR:.1f} mm/year")
            
            print(f"\n各驱动因子贡献率:")
            print(f"  气候变化 (CCV):      {attribution.C_CCV:.1f}%")
            print(f"  土地利用变化 (LUCC): {attribution.C_LUCC:.1f}%")
            print(f"  人类取用水 (WADR):    {attribution.C_WADR:.1f}%")
            
            print(f"\n弹性系数:")
            print(f"  εP (降水弹性):   {attribution.elasticity['epsilon_P']:.3f}")
            print(f"  εPET (PET弹性):  {attribution.elasticity['epsilon_PET']:.3f}")
            print(f"  εn (参数n弹性):  {attribution.elasticity['epsilon_n']:.3f}")
            
            print(f"\n【结论】")
            dominant_factor = max(
                [('气候变化', abs(attribution.C_CCV)),
                 ('土地利用变化', abs(attribution.C_LUCC)),
                 ('人类取用水', abs(attribution.C_WADR))],
                key=lambda x: x[1]
            )
            print(f"主导因子: {dominant_factor[0]} (贡献率 {dominant_factor[1]:.1f}%)")


def example_5_bootstrap_uncertainty():
    """
    示例5：Bootstrap不确定性估计
    
    通过重采样评估参数校准的统计不确定性。
    """
    print("\n" + "="*70)
    print("示例5：Bootstrap不确定性估计")
    print("="*70)
    
    calibrator = ParameterCalibrator()
    
    # 创建具有年际变率的时间序列
    np.random.seed(456)
    n_years = 40
    time_series = pd.DataFrame({
        'P': np.random.normal(900, 100, n_years),  # 较大的年际变率
        'PET': np.random.normal(1100, 120, n_years),
        'Q_n': np.random.normal(250, 60, n_years)
    })
    
    print(f"\n数据特征:")
    print(f"样本数: {n_years} 年")
    print(f"降水变异系数: {time_series['P'].std() / time_series['P'].mean():.2%}")
    print(f"PET变异系数: {time_series['PET'].std() / time_series['PET'].mean():.2%}")
    print(f"径流变异系数: {time_series['Q_n'].std() / time_series['Q_n'].mean():.2%}")
    
    # 点估计（使用全部数据）
    P_mean = time_series['P'].mean()
    PET_mean = time_series['PET'].mean()
    Q_n_mean = time_series['Q_n'].mean()
    
    result_point = calibrator.calibrate_single_station(
        "UncertaintyTest", P_mean, PET_mean, Q_n_mean
    )
    
    print(f"\n【点估计】")
    print(f"n = {result_point.n:.4f}")
    print(f"E = {result_point.E:.1f} mm/year")
    
    # Bootstrap不确定性估计
    print(f"\n执行Bootstrap重采样 (1000次)...")
    uncertainty = calibrator.bootstrap_uncertainty(
        time_series,
        n_bootstrap=1000,
        confidence_level=0.95
    )
    
    print(f"\n【95%置信区间】")
    for param, (lower, upper) in uncertainty.items():
        if param == 'n':
            point_value = result_point.n
        elif param == 'E':
            point_value = result_point.E
        elif param == 'Q_n':
            point_value = Q_n_mean
        
        print(f"{param}: [{lower:.2f}, {upper:.2f}]  "
              f"(点估计: {point_value:.2f}, CI宽度: {upper - lower:.2f})")
    
    print(f"\n【解释】")
    print(f"置信区间反映了由于数据年际变率导致的参数不确定性。")
    print(f"较窄的CI表明参数估计较为稳定。")


def example_6_ensemble_attribution():
    """
    示例6：区域集合归因统计
    
    综合多个站点的归因结果，提供区域尺度的平均归因。
    """
    print("\n" + "="*70)
    print("示例6：区域集合归因统计")
    print("="*70)
    
    from src.budyko_model.parameter_calibration import AttributionResult
    
    # 模拟10个站点的归因结果
    np.random.seed(789)
    attribution_results = []
    
    for i in range(10):
        station_id = f"Region_Station_{i+1:02d}"
        
        # 模拟不同站点的归因结果（带有随机扰动）
        C_CCV_base = 55.0 + np.random.normal(0, 10)
        C_LUCC_base = 25.0 + np.random.normal(0, 5)
        C_WADR_base = 20.0 + np.random.normal(0, 5)
        
        result = AttributionResult(
            station_id=station_id,
            delta_Q_obs=-45.0 + np.random.normal(0, 10),
            delta_Q_n=-35.0 + np.random.normal(0, 8),
            delta_Q_CCV=-25.0 + np.random.normal(0, 5),
            delta_Q_LUCC=-10.0 + np.random.normal(0, 3),
            delta_Q_WADR=-10.0 + np.random.normal(0, 3),
            C_CCV=C_CCV_base,
            C_LUCC=C_LUCC_base,
            C_WADR=C_WADR_base,
            period_base="1960-1985",
            period_impact="1986-2016"
        )
        attribution_results.append(result)
    
    print(f"\n区域内站点数: {len(attribution_results)}")
    
    # 显示各站点结果
    print("\n各站点归因结果:")
    print(f"{'站点':<20} {'C_CCV':>8} {'C_LUCC':>8} {'C_WADR':>8}")
    print("-" * 50)
    for result in attribution_results:
        print(f"{result.station_id:<20} {result.C_CCV:>7.1f}% {result.C_LUCC:>7.1f}% "
              f"{result.C_WADR:>7.1f}%")
    
    # 计算集合统计
    ensemble_mean = calculate_ensemble_attribution(attribution_results, method='mean')
    ensemble_median = calculate_ensemble_attribution(attribution_results, method='median')
    
    print(f"\n【区域集合统计 - 平均值】")
    print(f"气候变化 (CCV):      {ensemble_mean['C_CCV_mean']:.1f}% ± "
          f"{ensemble_mean['C_CCV_std']:.1f}%")
    print(f"土地利用变化 (LUCC): {ensemble_mean['C_LUCC_mean']:.1f}% ± "
          f"{ensemble_mean['C_LUCC_std']:.1f}%")
    print(f"人类取用水 (WADR):    {ensemble_mean['C_WADR_mean']:.1f}% ± "
          f"{ensemble_mean['C_WADR_std']:.1f}%")
    
    print(f"\n【区域集合统计 - 中位数】")
    print(f"气候变化 (CCV):      {ensemble_median['C_CCV_median']:.1f}%")
    print(f"土地利用变化 (LUCC): {ensemble_median['C_LUCC_median']:.1f}%")
    print(f"人类取用水 (WADR):    {ensemble_median['C_WADR_median']:.1f}%")
    
    print(f"\n【区域结论】")
    print(f"该区域内，气候变化是径流变化的主导因子 (~{ensemble_mean['C_CCV_mean']:.0f}%)，")
    print(f"其次是土地利用变化 (~{ensemble_mean['C_LUCC_mean']:.0f}%) 和人类取用水 "
          f"(~{ensemble_mean['C_WADR_mean']:.0f}%)。")


def example_7_complete_workflow():
    """
    示例7：完整工作流演示
    
    集成模块1-5，展示从数据加载到归因分析的完整流程。
    """
    print("\n" + "="*70)
    print("示例7：完整工作流演示（模拟）")
    print("="*70)
    
    print("\n本示例模拟以下工作流：")
    print("1. 加载GRDC观测径流数据 (模块1)")
    print("2. 提取ISIMIP气候数据 (P, PET) (模块2-3)")
    print("3. 计算天然径流 (Q_n = Q_obs + 耗水) (模块1)")
    print("4. 参数校准和归因分析 (模块4-5)")
    print("5. 结果导出和可视化")
    
    calibrator = ParameterCalibrator(change_point=1986, min_valid_years=15)
    
    # 步骤1: 模拟加载5个站点的数据
    print("\n【步骤1】加载站点数据...")
    stations = ['ST001', 'ST002', 'ST003', 'ST004', 'ST005']
    
    # 步骤2-3: 模拟时间序列数据
    print("【步骤2-3】提取气候数据和计算天然径流...")
    np.random.seed(2025)
    
    all_results = []
    
    for station_id in stations:
        # 创建1960-2016年时间序列
        years = np.arange(1960, 2017)
        
        time_series = pd.DataFrame({
            'year': years,
            'P': np.random.normal(850 + np.random.randint(-200, 200), 70, len(years)),
            'PET': np.random.normal(1100 + np.random.randint(-200, 200), 90, len(years)),
            'Q_n': np.random.normal(250 + np.random.randint(-100, 100), 40, len(years)),
            'Q_obs': np.random.normal(230 + np.random.randint(-100, 100), 40, len(years))
        })
        
        # 步骤4: 参数演变分析
        evolution_results = calibrator.analyze_parameter_evolution(
            station_id=station_id,
            time_series=time_series
        )
        
        if 'period_1' in evolution_results and 'period_2' in evolution_results:
            # 计算归因
            Q_obs_1 = time_series[time_series['year'] < 1986]['Q_obs'].mean()
            Q_obs_2 = time_series[time_series['year'] >= 1986]['Q_obs'].mean()
            
            attribution = calibrator.calculate_attribution(
                station_id=station_id,
                period_1_data=evolution_results['period_1'],
                period_2_data=evolution_results['period_2'],
                Q_obs_1=Q_obs_1,
                Q_obs_2=Q_obs_2
            )
            
            if attribution:
                all_results.append(attribution)
    
    print(f"【步骤4】参数校准和归因分析完成，成功处理 {len(all_results)}/{len(stations)} 个站点")
    
    # 步骤5: 结果汇总
    print("\n【步骤5】结果汇总和导出...")
    
    if all_results:
        results_df = pd.DataFrame([r.to_dict() for r in all_results])
        
        print("\n归因结果汇总表:")
        print(results_df[['station_id', 'delta_Q_obs', 'C_CCV', 'C_LUCC', 'C_WADR']].to_string(index=False))
        
        # 区域统计
        ensemble = calculate_ensemble_attribution(all_results, method='mean')
        
        print(f"\n区域平均归因:")
        print(f"  CCV:  {ensemble['C_CCV_mean']:.1f}% ± {ensemble['C_CCV_std']:.1f}%")
        print(f"  LUCC: {ensemble['C_LUCC_mean']:.1f}% ± {ensemble['C_LUCC_std']:.1f}%")
        print(f"  WADR: {ensemble['C_WADR_mean']:.1f}% ± {ensemble['C_WADR_std']:.1f}%")
        
        print("\n工作流演示完成！")
        print("实际应用中，将使用真实的GRDC、ISIMIP和Huang et al. (2018)数据。")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("模块5使用示例：批量站点参数校准和归因分析")
    print("="*70)
    
    examples = [
        ("单站点参数校准", example_1_single_station_calibration),
        ("批量多站点校准", example_2_batch_calibration),
        ("参数演变分析", example_3_parameter_evolution),
        ("完整归因分析", example_4_complete_attribution),
        ("Bootstrap不确定性估计", example_5_bootstrap_uncertainty),
        ("区域集合归因统计", example_6_ensemble_attribution),
        ("完整工作流演示", example_7_complete_workflow)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n示例{i} ({name}) 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("所有示例执行完毕")
    print("="*70)


if __name__ == "__main__":
    main()
