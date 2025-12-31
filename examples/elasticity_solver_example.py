"""
elasticity_solver_example.py

弹性系数计算模块使用示例

本示例展示如何使用elasticity_solver模块计算径流对各驱动因子的弹性系数，
包括：
1. 单站点弹性系数计算
2. 不同气候类型的弹性系数对比
3. 弹性系数的物理意义解读
4. 批量站点处理
5. 结果验证和质量控制

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

from src.budyko_model.elasticity_solver import (
    calculate_elasticity_P,
    calculate_elasticity_PET,
    calculate_elasticity_n,
    calculate_all_elasticities,
    validate_elasticity_signs,
    batch_calculate_elasticities
)


def example_1_single_station():
    """
    示例1：单站点弹性系数计算
    
    展示如何为单个流域计算三项弹性系数，并解读其物理意义。
    """
    print("\n" + "="*70)
    print("示例1：单站点弹性系数计算")
    print("="*70)
    
    # 长江上游某流域的气候数据
    P = 1050.0      # 年均降水 (mm)
    PET = 950.0     # 年均PET (mm)
    n = 2.35        # 校准得到的参数n
    
    print(f"\n流域特征:")
    print(f"  年均降水: P = {P:.1f} mm")
    print(f"  年均PET: PET = {PET:.1f} mm")
    print(f"  景观参数: n = {n:.3f}")
    print(f"  干旱指数: φ = PET/P = {PET/P:.3f} (湿润型)")
    
    # 计算各项弹性系数
    eps_P = calculate_elasticity_P(P, PET, n)
    eps_PET = calculate_elasticity_PET(P, PET, n)
    eps_n = calculate_elasticity_n(P, PET, n)
    
    print(f"\n弹性系数计算结果:")
    print(f"  εP = {eps_P:.3f}")
    print(f"  εPET = {eps_PET:.3f}")
    print(f"  εn = {eps_n:.3f}")
    print(f"  εP + εPET = {eps_P + eps_PET:.3f}")
    
    # 物理意义解读
    print(f"\n物理意义解读:")
    print(f"  • 降水增加1% → 径流增加{eps_P:.2f}% (放大效应)")
    print(f"  • PET增加1% → 径流减少{abs(eps_PET):.2f}%")
    print(f"  • 参数n增加1% → 径流减少{abs(eps_n):.2f}%")
    
    # 验证符号合理性
    is_valid, msg = validate_elasticity_signs(eps_P, eps_PET, eps_n)
    print(f"\n验证结果: {msg}")


def example_2_climate_comparison():
    """
    示例2：不同气候类型的弹性系数对比
    
    对比湿润、半湿润、半干旱、干旱四种气候类型的弹性系数差异。
    """
    print("\n" + "="*70)
    print("示例2：不同气候类型的弹性系数对比")
    print("="*70)
    
    # 定义四种气候类型的代表流域
    climates = {
        '湿润型': {'P': 1500, 'PET': 800, 'n': 2.8},
        '半湿润': {'P': 800, 'PET': 1000, 'n': 2.3},
        '半干旱': {'P': 500, 'PET': 1200, 'n': 1.8},
        '干旱型': {'P': 300, 'PET': 1500, 'n': 1.2}
    }
    
    results = []
    for climate_type, params in climates.items():
        P, PET, n = params['P'], params['PET'], params['n']
        phi = PET / P
        
        elasticities = calculate_all_elasticities(P, PET, n)
        
        results.append({
            '气候类型': climate_type,
            'P (mm)': P,
            'PET (mm)': PET,
            'φ': phi,
            'n': n,
            'εP': elasticities['epsilon_P'],
            'εPET': elasticities['epsilon_PET'],
            'εn': elasticities['epsilon_n']
        })
    
    df = pd.DataFrame(results)
    
    print("\n弹性系数对比:")
    print(df.to_string(index=False))
    
    print("\n主要发现:")
    print("  • εP随干旱程度增加而增大（干旱区径流对降水更敏感）")
    print("  • |εPET|随干旱程度增加而增大")
    print("  • εn的变化相对较小，主要取决于n值本身")


def example_3_sensitivity_analysis():
    """
    示例3：参数n的敏感性分析
    
    研究参数n变化对弹性系数的影响。
    """
    print("\n" + "="*70)
    print("示例3：参数n的敏感性分析")
    print("="*70)
    
    # 固定气候条件
    P = 800
    PET = 1200
    
    # 变化的参数n（从1.0到4.0）
    n_values = np.linspace(1.0, 4.0, 13)
    
    eps_P_list = []
    eps_PET_list = []
    eps_n_list = []
    
    for n in n_values:
        elasticities = calculate_all_elasticities(P, PET, n)
        eps_P_list.append(elasticities['epsilon_P'])
        eps_PET_list.append(elasticities['epsilon_PET'])
        eps_n_list.append(elasticities['epsilon_n'])
    
    print(f"\n固定条件: P={P} mm, PET={PET} mm")
    print("\nn值变化对弹性系数的影响:")
    print("-" * 60)
    print(f"{'n':>6}  {'εP':>8}  {'εPET':>8}  {'εn':>8}  {'εP+εPET':>10}")
    print("-" * 60)
    
    for i, n in enumerate(n_values):
        eps_sum = eps_P_list[i] + eps_PET_list[i]
        print(f"{n:6.2f}  {eps_P_list[i]:8.3f}  {eps_PET_list[i]:8.3f}  "
              f"{eps_n_list[i]:8.3f}  {eps_sum:10.3f}")
    
    print("\n观察:")
    print("  • n增大时，εP略有下降")
    print("  • n增大时，|εPET|略有下降")
    print("  • εP + εPET随n变化，但保持在合理范围内")


def example_4_batch_processing():
    """
    示例4：批量站点处理
    
    展示如何对多个站点批量计算弹性系数。
    """
    print("\n" + "="*70)
    print("示例4：批量站点处理")
    print("="*70)
    
    # 创建10个站点的数据
    np.random.seed(42)
    
    stations_data = pd.DataFrame({
        'station_id': [f'S{i:03d}' for i in range(1, 11)],
        'basin': ['长江流域', '黄河流域', '珠江流域', '淮河流域', '海河流域',
                 '松花江', '辽河流域', '塔里木河', '黑河流域', '青海湖'],
        'P': np.random.uniform(300, 1500, 10),
        'PET': np.random.uniform(600, 1800, 10),
        'n': np.random.uniform(1.2, 3.5, 10)
    })
    
    print("\n站点数据（前5行）:")
    print(stations_data[['station_id', 'basin', 'P', 'PET', 'n']].head())
    
    # 批量计算弹性系数
    results = batch_calculate_elasticities(stations_data)
    
    print("\n弹性系数计算结果（前5行）:")
    print(results[['station_id', 'epsilon_P', 'epsilon_PET', 'epsilon_n']].head())
    
    # 统计分析
    print("\n统计摘要:")
    print(results[['epsilon_P', 'epsilon_PET', 'epsilon_n']].describe())
    
    # 验证所有站点的符号
    valid_count = 0
    for _, row in results.iterrows():
        is_valid, _ = validate_elasticity_signs(
            row['epsilon_P'], row['epsilon_PET'], row['epsilon_n']
        )
        if is_valid:
            valid_count += 1
    
    print(f"\n符号验证: {valid_count}/{len(results)} 个站点通过验证")


def example_5_elasticity_application():
    """
    示例5：弹性系数在归因分析中的应用
    
    演示如何利用弹性系数进行径流变化归因。
    """
    print("\n" + "="*70)
    print("示例5：弹性系数在归因分析中的应用")
    print("="*70)
    
    # 基准期（1960-1985）的气候条件
    P_base = 650.0
    PET_base = 900.0
    n_base = 2.0
    Q_base = 200.0  # 基准期平均径流
    
    # 影响期（1986-2016）的气候条件
    P_impact = 620.0     # 降水减少
    PET_impact = 950.0   # PET增加
    n_impact = 2.3       # n增大（植被恢复）
    Q_impact = 150.0     # 观测径流减少
    
    print("\n基准期（1960-1985）:")
    print(f"  P = {P_base:.1f} mm, PET = {PET_base:.1f} mm")
    print(f"  n = {n_base:.2f}, Q_obs = {Q_base:.1f} mm")
    
    print("\n影响期（1986-2016）:")
    print(f"  P = {P_impact:.1f} mm, PET = {PET_impact:.1f} mm")
    print(f"  n = {n_impact:.2f}, Q_obs = {Q_impact:.1f} mm")
    
    # 计算变化量
    delta_P = P_impact - P_base
    delta_PET = PET_impact - PET_base
    delta_n = n_impact - n_base
    delta_Q = Q_impact - Q_base
    
    print("\n变化量:")
    print(f"  ΔP = {delta_P:.1f} mm ({delta_P/P_base*100:.1f}%)")
    print(f"  ΔPET = {delta_PET:.1f} mm ({delta_PET/PET_base*100:.1f}%)")
    print(f"  Δn = {delta_n:.3f} ({delta_n/n_base*100:.1f}%)")
    print(f"  ΔQ_obs = {delta_Q:.1f} mm ({delta_Q/Q_base*100:.1f}%)")
    
    # 使用全时段均值计算弹性系数
    P_mean = (P_base + P_impact) / 2
    PET_mean = (PET_base + PET_impact) / 2
    n_mean = (n_base + n_impact) / 2
    Q_mean = (Q_base + Q_impact) / 2
    
    elasticities = calculate_all_elasticities(P_mean, PET_mean, n_mean)
    
    print(f"\n弹性系数（基于全时段均值）:")
    print(f"  εP = {elasticities['epsilon_P']:.3f}")
    print(f"  εPET = {elasticities['epsilon_PET']:.3f}")
    print(f"  εn = {elasticities['epsilon_n']:.3f}")
    
    # 归因计算（根据main.tex方程7）
    delta_Q_CCV = (elasticities['epsilon_P'] * Q_mean / P_mean * delta_P +
                   elasticities['epsilon_PET'] * Q_mean / PET_mean * delta_PET)
    
    delta_Q_LUCC = elasticities['epsilon_n'] * Q_mean / n_mean * delta_n
    
    # 计算贡献率
    C_CCV = delta_Q_CCV / delta_Q * 100
    C_LUCC = delta_Q_LUCC / delta_Q * 100
    
    print(f"\n归因分析结果:")
    print(f"  气候变化贡献: ΔQ_CCV = {delta_Q_CCV:.1f} mm ({C_CCV:.1f}%)")
    print(f"    - 降水变化: {elasticities['epsilon_P'] * Q_mean / P_mean * delta_P:.1f} mm")
    print(f"    - PET变化: {elasticities['epsilon_PET'] * Q_mean / PET_mean * delta_PET:.1f} mm")
    print(f"  土地利用变化贡献: ΔQ_LUCC = {delta_Q_LUCC:.1f} mm ({C_LUCC:.1f}%)")
    
    print(f"\n结论:")
    if C_CCV > C_LUCC:
        print(f"  • 气候变化是径流减少的主导因素（{C_CCV:.0f}%）")
    else:
        print(f"  • 土地利用变化是径流减少的主导因素（{C_LUCC:.0f}%）")


def example_6_quality_control():
    """
    示例6：弹性系数的质量控制
    
    展示如何检验弹性系数的合理性，识别异常结果。
    """
    print("\n" + "="*70)
    print("示例6：弹性系数的质量控制")
    print("="*70)
    
    # 创建包含正常和异常站点的测试数据
    test_cases = [
        # (名称, P, PET, n, 预期状态)
        ("正常站点1", 800, 1200, 2.5, "正常"),
        ("正常站点2", 1200, 900, 2.8, "正常"),
        ("极端干旱", 200, 2000, 1.2, "正常"),
        ("极端湿润", 2000, 500, 3.5, "正常"),
    ]
    
    print("\n质量控制检查:")
    print("-" * 80)
    
    for name, P, PET, n, expected in test_cases:
        elasticities = calculate_all_elasticities(P, PET, n)
        
        # 基本符号检查
        is_valid_basic, msg_basic = validate_elasticity_signs(
            elasticities['epsilon_P'],
            elasticities['epsilon_PET'],
            elasticities['epsilon_n']
        )
        
        # 严格检查
        is_valid_strict, msg_strict = validate_elasticity_signs(
            elasticities['epsilon_P'],
            elasticities['epsilon_PET'],
            elasticities['epsilon_n'],
            strict=True
        )
        
        print(f"\n{name} (P={P}, PET={PET}, n={n}):")
        print(f"  εP = {elasticities['epsilon_P']:.3f}")
        print(f"  εPET = {elasticities['epsilon_PET']:.3f}")
        print(f"  εn = {elasticities['epsilon_n']:.3f}")
        print(f"  基本检查: {'✓' if is_valid_basic else '✗'} {msg_basic}")
        print(f"  严格检查: {'✓' if is_valid_strict else '✗'} {msg_strict}")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("弹性系数计算模块 (elasticity_solver) 使用示例")
    print("="*70)
    
    example_1_single_station()
    example_2_climate_comparison()
    example_3_sensitivity_analysis()
    example_4_batch_processing()
    example_5_elasticity_application()
    example_6_quality_control()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
