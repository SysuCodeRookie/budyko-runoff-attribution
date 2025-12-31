"""
ISIMIP归因分析使用示例

本脚本展示如何使用ISIMIPAttribution类进行径流归因分析，
包括多模型集成、ACC/NCV分离、不确定性分析等高级功能。

示例包括：
1. 单站点ISIMIP归因分析
2. 多模型不确定性评估
3. ACC与NCV对比分析
4. 与Budyko方法的一致性验证
5. 批量多站点处理

作者: Research Software Engineer
日期: 2025-01-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.budyko_model.isimip_attribution import (
    ISIMIPAttribution,
    batch_isimip_attribution,
    summarize_isimip_attribution,
)
from src.budyko_model.budyko_attribution import BudykoAttribution


# ============================================================================
# 示例数据生成（模拟真实ISIMIP数据结构）
# ============================================================================

def generate_sample_isimip_data():
    """
    生成示例ISIMIP数据（模拟真实场景）
    
    场景说明：
    - 观测径流（Q_o）: 200mm → 150mm（-25%，显著减少）
    - 天然径流（Q_n）: 250mm → 220mm（-12%，部分还原）
    
    ISIMIP模型模拟：
    - Q'_o（obsclim+histsoc）: 模拟观测，包含所有影响
    - Q'_n（obsclim+1901soc）: 模拟天然化，仅气候变化
    - Q'_cn（counterclim+1901soc）: 去趋势基准，仅自然变率
    """
    years = range(1960, 2017)
    
    # 观测数据（站点）
    obs_data = []
    for year in years:
        if year < 1986:
            Q_o = 200 + np.random.normal(0, 10)
            Q_n = 250 + np.random.normal(0, 10)
        else:
            Q_o = 150 + np.random.normal(0, 10)  # 观测径流减少50mm
            Q_n = 220 + np.random.normal(0, 10)  # 天然径流减少30mm
        
        obs_data.append({
            'year': year,
            'Q_o': max(Q_o, 10),
            'Q_n': max(Q_n, 10)
        })
    
    obs_df = pd.DataFrame(obs_data)
    
    # ISIMIP模型列表（9个GHMs）
    models = ['clm45', 'cwatm', 'h08', 'lpjml', 'matsiro', 
              'mpi-hm', 'pcr-globwb', 'vic', 'watergap2']
    
    # 场景1: obsclim + histsoc (Q'_o)
    # 应接近观测Q_o
    scenario1_data = {'year': list(years)}
    for i, model in enumerate(models):
        model_values = []
        for year in years:
            if year < 1986:
                base = 200 + i * 5  # 模型间差异
            else:
                base = 150 + i * 5
            model_values.append(base + np.random.normal(0, 8))
        scenario1_data[model] = model_values
    
    # 场景2: obsclim + 1901soc (Q'_n)
    # 天然化径流，气候变化影响（减少约25mm）
    scenario2_data = {'year': list(years)}
    for i, model in enumerate(models):
        model_values = []
        for year in years:
            if year < 1986:
                base = 250 + i * 5
            else:
                base = 225 + i * 5  # 气候变化导致减少25mm
            model_values.append(base + np.random.normal(0, 8))
        scenario2_data[model] = model_values
    
    # 场景3: counterclim + 1901soc (Q'_cn)
    # 去趋势基准，仅自然变率（变化很小）
    scenario3_data = {'year': list(years)}
    for i, model in enumerate(models):
        model_values = []
        for year in years:
            # 去除人为气候变化后，径流几乎不变（仅5mm减少）
            if year < 1986:
                base = 250 + i * 5
            else:
                base = 245 + i * 5  # 自然变率导致轻微减少
            model_values.append(base + np.random.normal(0, 8))
        scenario3_data[model] = model_values
    
    isimip_data = {
        'obsclim_histsoc': pd.DataFrame(scenario1_data),
        'obsclim_1901soc': pd.DataFrame(scenario2_data),
        'counterclim_1901soc': pd.DataFrame(scenario3_data)
    }
    
    return obs_df, isimip_data, models


# ============================================================================
# 示例1: 单站点ISIMIP归因分析
# ============================================================================

def example1_single_station_attribution():
    """
    示例1: 单站点ISIMIP归因分析
    
    展示如何：
    1. 加载ISIMIP多模型数据
    2. 执行归因分析
    3. 解读结果
    """
    print("=" * 70)
    print("示例1: 单站点ISIMIP归因分析")
    print("=" * 70)
    print()
    
    # 生成示例数据
    obs_data, isimip_data, models = generate_sample_isimip_data()
    
    # 创建归因对象
    attribution = ISIMIPAttribution(obs_data, isimip_data)
    
    # 设置时段（1986年为突变点）
    attribution.set_periods(change_year=1986)
    
    print("数据加载完成:")
    print(f"  观测数据: 1960-2016 ({len(obs_data)}年)")
    print(f"  ISIMIP模型数: {len(models)}")
    print(f"  场景数: 3 (obsclim+histsoc, obsclim+1901soc, counterclim+1901soc)")
    print()
    
    # 执行归因
    print("执行ISIMIP归因分析...")
    results = attribution.run_attribution()
    print("✓ 归因完成")
    print()
    
    # 输出结果
    print("【观测径流变化】")
    print(f"  基准期 (1960-1985): Q_o = {results['Q_o_pre']:.1f} mm")
    print(f"  影响期 (1986-2016): Q_o = {results['Q_o_post']:.1f} mm")
    print(f"  变化量: ΔQ_o = {results['delta_Q_o']:.1f} mm ({results['delta_Q_o']/results['Q_o_pre']*100:+.1f}%)")
    print()
    
    print("【ISIMIP多模型集成结果】")
    print(f"  Q'_o变化: {results['delta_Q_prime_o']:.1f} mm")
    print(f"  Q'_n变化: {results['delta_Q_prime_n']:.1f} mm")
    print(f"  Q'_cn变化: {results['delta_Q_prime_cn']:.1f} mm")
    print()
    
    print("【归因结果 - 三因子分解】")
    print(f"  气候变化与变率 (C_CCV): {results['C_CCV_isimip']:+.1f}%")
    print(f"  土地利用变化 (C_LUCC): {results['C_LUCC_isimip']:+.1f}%")
    print(f"  人类取用水 (C_WADR): {results['C_WADR_isimip']:+.1f}%")
    print(f"  ────────────────────────────")
    print(f"  贡献率之和: {results['contribution_sum']:.1f}%")
    print()
    
    print("【气候变化信号分离】")
    print(f"  人为气候变化 (C_ACC): {results['C_ACC']:+.1f}%")
    print(f"  自然气候变率 (C_NCV): {results['C_NCV']:+.1f}%")
    print(f"  ────────────────────────────")
    print(f"  C_ACC + C_NCV: {results['ACC_NCV_sum']:.1f}% (应≈C_CCV)")
    print()
    
    # 确定主导因素
    contributions = {
        'ACC': abs(results['C_ACC']),
        'NCV': abs(results['C_NCV']),
        'LUCC': abs(results['C_LUCC_isimip']),
        'WADR': abs(results['C_WADR_isimip'])
    }
    dominant = max(contributions, key=contributions.get)
    print(f"【主导因素】: {dominant} ({contributions[dominant]:.1f}%)")
    print()
    
    return results


# ============================================================================
# 示例2: 多模型不确定性评估
# ============================================================================

def example2_model_uncertainty(obs_data, isimip_data):
    """
    示例2: 多模型不确定性评估
    
    展示如何：
    1. 计算多模型统计量
    2. 评估模型间差异
    3. 可视化不确定性
    """
    print("=" * 70)
    print("示例2: 多模型不确定性评估")
    print("=" * 70)
    print()
    
    attribution = ISIMIPAttribution(obs_data, isimip_data)
    attribution.set_periods(change_year=1986)
    
    # 计算各场景的不确定性
    scenarios = ['obsclim_histsoc', 'obsclim_1901soc', 'counterclim_1901soc']
    
    print("【基准期 (1960-1985) 多模型统计】")
    for scenario in scenarios:
        unc = attribution.calculate_model_uncertainty(scenario, (1960, 1985))
        print(f"\n场景: {scenario}")
        print(f"  集合平均: {unc['mean']:.1f} mm")
        print(f"  标准差: ±{unc['std']:.1f} mm")
        print(f"  范围: [{unc['min']:.1f}, {unc['max']:.1f}] mm")
        print(f"  四分位距: [{unc['q25']:.1f}, {unc['q75']:.1f}] mm")
    
    print("\n" + "=" * 70)
    print()
    
    # 可视化不确定性（箱线图）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_data = []
    labels = []
    for scenario in scenarios:
        # 获取各模型的基准期平均值
        df = isimip_data[scenario]
        df_pre = df[(df['year'] >= 1960) & (df['year'] <= 1985)]
        model_means = df_pre[attribution.models].mean(axis=0).values
        box_data.append(model_means)
        
        # 简化标签
        if scenario == 'obsclim_histsoc':
            labels.append("Q'_o\n(obsclim+histsoc)")
        elif scenario == 'obsclim_1901soc':
            labels.append("Q'_n\n(obsclim+1901soc)")
        else:
            labels.append("Q'_cn\n(counterclim+1901soc)")
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    # 美化箱线图
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('径流 (mm/year)', fontsize=12)
    ax.set_title('ISIMIP多模型不确定性 (基准期 1960-1985)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/isimip_model_uncertainty.png', dpi=300, bbox_inches='tight')
    print("✓ 不确定性箱线图已保存: results/isimip_model_uncertainty.png")
    print()


# ============================================================================
# 示例3: ACC与NCV对比分析
# ============================================================================

def example3_ACC_NCV_comparison(results):
    """
    示例3: ACC与NCV对比分析
    
    展示如何：
    1. 解读ACC和NCV的物理意义
    2. 评估人为气候变化的影响
    3. 可视化贡献对比
    """
    print("=" * 70)
    print("示例3: 人为气候变化(ACC) vs 自然变率(NCV)")
    print("=" * 70)
    print()
    
    print("【概念解释】")
    print("  • ACC (Anthropogenic Climate Change): 人为气候变化")
    print("    - 由温室气体排放等人类活动导致的气候系统变化")
    print("    - 计算: C_ACC = (ΔQ'_n - ΔQ'_cn) / ΔQ_o × 100%")
    print()
    print("  • NCV (Natural Climate Variability): 自然气候变率")
    print("    - 气候系统的自然波动（如ENSO、PDO）")
    print("    - 计算: C_NCV = ΔQ'_cn / ΔQ_o × 100%")
    print()
    
    C_ACC = results['C_ACC']
    C_NCV = results['C_NCV']
    C_CCV = results['C_CCV_isimip']
    
    print("【本站点结果】")
    print(f"  C_ACC = {C_ACC:+.1f}%")
    print(f"  C_NCV = {C_NCV:+.1f}%")
    print(f"  C_CCV = {C_CCV:+.1f}% (ACC + NCV)")
    print()
    
    # 判断ACC/NCV的相对重要性
    if abs(C_ACC) > abs(C_NCV):
        print("  → 人为气候变化是气候信号的主导因素")
        print(f"    ACC贡献是NCV的 {abs(C_ACC)/abs(C_NCV):.1f} 倍")
    else:
        print("  → 自然气候变率是气候信号的主导因素")
        print(f"    NCV贡献是ACC的 {abs(C_NCV)/abs(C_ACC):.1f} 倍")
    print()
    
    # 可视化ACC vs NCV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1: ACC和NCV堆叠柱状图
    categories = ['气候变化总贡献\n(C_CCV)']
    acc_values = [C_ACC]
    ncv_values = [C_NCV]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x, acc_values, width, label='人为气候变化 (ACC)', color='#d62728')
    ax1.bar(x, ncv_values, width, bottom=acc_values, label='自然气候变率 (NCV)', color='#1f77b4')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax1.set_ylabel('贡献率 (%)', fontsize=12)
    ax1.set_title('ACC与NCV分解', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 子图2: 饼图显示ACC/NCV占比
    sizes = [abs(C_ACC), abs(C_NCV)]
    labels_pie = [f'ACC\n({abs(C_ACC):.1f}%)', f'NCV\n({abs(C_NCV):.1f}%)']
    colors_pie = ['#d62728', '#1f77b4']
    
    ax2.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('气候变化信号构成', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/isimip_ACC_NCV_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ ACC/NCV对比图已保存: results/isimip_ACC_NCV_comparison.png")
    print()


# ============================================================================
# 示例4: 与Budyko方法一致性验证
# ============================================================================

def example4_budyko_comparison():
    """
    示例4: ISIMIP归因与Budyko归因的一致性验证
    
    展示如何：
    1. 同时运行两种归因方法
    2. 对比结果差异
    3. 理解差异来源
    """
    print("=" * 70)
    print("示例4: ISIMIP归因 vs Budyko归因对比")
    print("=" * 70)
    print()
    
    # 生成数据
    obs_data, isimip_data, models = generate_sample_isimip_data()
    
    # 添加Budyko所需的P, PET列（模拟）
    obs_data_budyko = obs_data.copy()
    obs_data_budyko['P'] = 800 - (obs_data_budyko['year'] >= 1986) * 50  # 降水减少
    obs_data_budyko['PET'] = 1000 + (obs_data_budyko['year'] >= 1986) * 100  # PET增加
    obs_data_budyko['Qn'] = obs_data_budyko['Q_n']
    obs_data_budyko['Qo'] = obs_data_budyko['Q_o']
    
    # Budyko归因
    print("执行Budyko归因...")
    budyko_attr = BudykoAttribution(obs_data_budyko)
    budyko_attr.set_periods(change_year=1986)
    budyko_results = budyko_attr.run_attribution()
    print("✓ Budyko归因完成")
    print()
    
    # ISIMIP归因
    print("执行ISIMIP归因...")
    isimip_attr = ISIMIPAttribution(obs_data, isimip_data)
    isimip_attr.set_periods(change_year=1986)
    isimip_results = isimip_attr.run_attribution()
    print("✓ ISIMIP归因完成")
    print()
    
    # 对比结果
    print("【归因结果对比】")
    print(f"{'因子':<15} {'Budyko (%)':<15} {'ISIMIP (%)':<15} {'差异 (%)':<15}")
    print("-" * 60)
    
    ccv_diff = isimip_results['C_CCV_isimip'] - budyko_results['C_CCV']
    lucc_diff = isimip_results['C_LUCC_isimip'] - budyko_results['C_LUCC']
    wadr_diff = isimip_results['C_WADR_isimip'] - budyko_results['C_WADR']
    
    print(f"{'CCV':<15} {budyko_results['C_CCV']:>+14.1f} {isimip_results['C_CCV_isimip']:>+14.1f} {ccv_diff:>+14.1f}")
    print(f"{'LUCC':<15} {budyko_results['C_LUCC']:>+14.1f} {isimip_results['C_LUCC_isimip']:>+14.1f} {lucc_diff:>+14.1f}")
    print(f"{'WADR':<15} {budyko_results['C_WADR']:>+14.1f} {isimip_results['C_WADR_isimip']:>+14.1f} {wadr_diff:>+14.1f}")
    print()
    
    # 计算对比统计
    comparison = isimip_attr.compare_with_budyko(budyko_results)
    
    print("【一致性统计】")
    print(f"  RMSE (均方根误差): {comparison['rmse']:.2f}%")
    print(f"  MAE (平均绝对误差): {comparison['mae']:.2f}%")
    print()
    
    if comparison['rmse'] < 10:
        print("  ✓ 两种方法结果高度一致（RMSE < 10%）")
    elif comparison['rmse'] < 20:
        print("  ◐ 两种方法结果基本一致（RMSE < 20%）")
    else:
        print("  ✗ 两种方法结果存在较大差异（RMSE ≥ 20%）")
    print()
    
    print("【差异来源分析】")
    print("  可能原因：")
    print("  1. Budyko假设的简化（稳态、参数化）")
    print("  2. ISIMIP模型的系统性偏差")
    print("  3. LUCC表征方式差异（参数n vs 直接模拟）")
    print("  4. 时间尺度匹配问题")
    print()
    
    # 可视化对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = ['CCV', 'LUCC', 'WADR']
    budyko_values = [budyko_results['C_CCV'], budyko_results['C_LUCC'], budyko_results['C_WADR']]
    isimip_values = [isimip_results['C_CCV_isimip'], isimip_results['C_LUCC_isimip'], isimip_results['C_WADR_isimip']]
    
    x = np.arange(len(factors))
    width = 0.35
    
    ax.bar(x - width/2, budyko_values, width, label='Budyko方法', color='#2ca02c')
    ax.bar(x + width/2, isimip_values, width, label='ISIMIP方法', color='#ff7f0e')
    
    ax.set_ylabel('贡献率 (%)', fontsize=12)
    ax.set_title('Budyko归因 vs ISIMIP归因对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/isimip_budyko_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 对比图已保存: results/isimip_budyko_comparison.png")
    print()


# ============================================================================
# 示例5: 批量多站点处理
# ============================================================================

def example5_batch_processing():
    """
    示例5: 批量多站点ISIMIP归因
    
    展示如何：
    1. 批量处理多个站点
    2. 汇总区域统计
    3. 识别空间模式
    """
    print("=" * 70)
    print("示例5: 批量多站点ISIMIP归因")
    print("=" * 70)
    print()
    
    # 生成3个站点的数据
    stations = ['STATION_A', 'STATION_B', 'STATION_C']
    all_obs = []
    isimip_dict = {}
    
    for i, station in enumerate(stations):
        obs, isimip, models = generate_sample_isimip_data()
        
        # 添加站点标识
        obs['station_id'] = station
        all_obs.append(obs)
        
        # 存储ISIMIP数据
        isimip_dict[station] = isimip
    
    # 合并观测数据
    multi_station_obs = pd.concat(all_obs, ignore_index=True)
    
    print(f"数据准备完成: {len(stations)}个站点")
    print()
    
    # 批量归因
    print("执行批量归因...")
    batch_results = batch_isimip_attribution(
        multi_station_obs,
        isimip_dict,
        change_year=1986
    )
    print("✓ 批量归因完成")
    print()
    
    # 显示结果
    print("【批量归因结果】")
    print(batch_results[['station_id', 'C_CCV_isimip', 'C_ACC', 'C_NCV', 
                          'C_LUCC_isimip', 'C_WADR_isimip']].to_string(index=False))
    print()
    
    # 区域统计
    print("【区域统计】")
    print(f"  C_CCV平均: {batch_results['C_CCV_isimip'].mean():.1f}% (±{batch_results['C_CCV_isimip'].std():.1f}%)")
    print(f"  C_ACC平均: {batch_results['C_ACC'].mean():.1f}% (±{batch_results['C_ACC'].std():.1f}%)")
    print(f"  C_NCV平均: {batch_results['C_NCV'].mean():.1f}% (±{batch_results['C_NCV'].std():.1f}%)")
    print(f"  C_LUCC平均: {batch_results['C_LUCC_isimip'].mean():.1f}% (±{batch_results['C_LUCC_isimip'].std():.1f}%)")
    print(f"  C_WADR平均: {batch_results['C_WADR_isimip'].mean():.1f}% (±{batch_results['C_WADR_isimip'].std():.1f}%)")
    print()
    
    # 保存结果
    batch_results.to_csv('results/batch_isimip_attribution.csv', index=False)
    print("✓ 批量结果已保存: results/batch_isimip_attribution.csv")
    print()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    import os
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 70)
    print(" ISIMIP归因分析使用示例")
    print("=" * 70 + "\n")
    
    # 示例1: 单站点归因
    results = example1_single_station_attribution()
    
    # 示例2: 不确定性评估
    obs_data, isimip_data, models = generate_sample_isimip_data()
    example2_model_uncertainty(obs_data, isimip_data)
    
    # 示例3: ACC vs NCV
    example3_ACC_NCV_comparison(results)
    
    # 示例4: 与Budyko对比
    example4_budyko_comparison()
    
    # 示例5: 批量处理
    example5_batch_processing()
    
    print("=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)
    print()
    print("生成的文件:")
    print("  • results/isimip_model_uncertainty.png")
    print("  • results/isimip_ACC_NCV_comparison.png")
    print("  • results/isimip_budyko_comparison.png")
    print("  • results/batch_isimip_attribution.csv")
    print()
