"""
Budyko核心方程使用示例

演示BudykoModel类的各种实际应用场景：
1. 基本蒸散发和径流计算
2. 参数n的校准
3. 弹性系数计算
4. 归因分析
5. 与前置模块（GRDC、气候数据、PET）的集成

作者: Budyko归因分析系统开发团队
日期: 2025-01-01
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.budyko_model.core_equations import (
    BudykoModel,
    validate_water_balance,
    calculate_aridity_index,
    estimate_n_from_climate
)


def example_1_basic_budyko_calculation():
    """
    示例1: 基本Budyko方程计算
    
    已知流域的P、PET和n，计算E和Q_n
    """
    print("=" * 60)
    print("示例1: 基本Budyko方程计算")
    print("=" * 60)
    
    # 初始化模型
    model = BudykoModel()
    
    # 假设某流域的气候条件
    P = 850  # mm/year
    PET = 1200  # mm/year
    n = 2.3  # 流域景观参数
    
    # 计算实际蒸散发
    E = model.calculate_actual_ET(P, PET, n)
    print(f"\n流域条件:")
    print(f"  降水量 P = {P} mm/year")
    print(f"  潜在蒸散发 PET = {PET} mm/year")
    print(f"  景观参数 n = {n}")
    print(f"  干旱指数 φ = PET/P = {PET/P:.2f}")
    
    print(f"\n计算结果:")
    print(f"  实际蒸散发 E = {E:.2f} mm/year")
    print(f"  蒸发比 E/P = {E/P:.2f}")
    
    # 计算天然径流
    Q_n = model.calculate_naturalized_runoff(P, PET, n)
    print(f"  天然径流 Q_n = {Q_n:.2f} mm/year")
    print(f"  径流系数 Q_n/P = {Q_n/P:.2f}")
    
    # 验证水量平衡
    print(f"\n水量平衡检验:")
    print(f"  P - E = {P - E:.2f} mm/year")
    print(f"  Q_n = {Q_n:.2f} mm/year")
    print(f"  误差 = {abs((P - E) - Q_n):.6f} mm/year")


def example_2_parameter_calibration():
    """
    示例2: 参数n校准
    
    已知观测的P、PET和Q_n，反算参数n
    这是Budyko归因分析的第一步（main.tex Step 1）
    """
    print("\n\n" + "=" * 60)
    print("示例2: 参数n校准（模型率定）")
    print("=" * 60)
    
    model = BudykoModel()
    
    # 某流域1960-2016年多年平均值（来自观测数据）
    P = 820  # mm/year
    PET = 1150  # mm/year（由气候数据计算得到）
    Q_n = 235  # mm/year（由GRDC数据 + 用水还原得到）
    
    print(f"\n观测数据（1960-2016年均值）:")
    print(f"  P = {P} mm/year")
    print(f"  PET = {PET} mm/year")
    print(f"  Q_n = {Q_n} mm/year")
    
    # 物理一致性检查
    is_valid, msg = validate_water_balance(P, Q_n)
    print(f"\n水量平衡检查: {msg}")
    
    if not is_valid:
        print("警告：数据未通过物理一致性检查！")
        return
    
    # 反演参数n
    print("\n正在校准参数n...")
    n = model.calibrate_parameter_n(P, PET, Q_n)
    
    print(f"\n校准结果:")
    print(f"  n = {n:.3f}")
    
    # 验证校准结果
    Q_n_simulated = model.calculate_naturalized_runoff(P, PET, n)
    print(f"\n验证:")
    print(f"  观测径流 Q_n_obs = {Q_n:.2f} mm/year")
    print(f"  模拟径流 Q_n_sim = {Q_n_simulated:.2f} mm/year")
    print(f"  相对误差 = {abs(Q_n_simulated - Q_n) / Q_n * 100:.4f}%")
    
    # 解释n值的物理意义
    print(f"\n参数n的物理解释:")
    if n > 2.5:
        print(f"  n = {n:.2f} > 2.5: 流域截留能力强（植被茂密、土壤深厚）")
    elif n > 1.5:
        print(f"  n = {n:.2f} ∈ [1.5, 2.5]: 中等流域特性")
    else:
        print(f"  n = {n:.2f} < 1.5: 产流能力强（裸地、城市化）")


def example_3_elasticity_analysis():
    """
    示例3: 径流弹性系数计算
    
    计算径流对P、PET和n的敏感性（main.tex Step 2）
    """
    print("\n\n" + "=" * 60)
    print("示例3: 径流弹性系数分析")
    print("=" * 60)
    
    model = BudykoModel()
    
    # 使用示例2校准的参数
    P = 820
    PET = 1150
    n = 2.234  # 假设这是校准得到的值
    
    # 计算弹性系数
    elasticities = model.calculate_elasticities(P, PET, n)
    
    print(f"\n流域条件:")
    print(f"  P = {P} mm/year, PET = {PET} mm/year, n = {n:.3f}")
    print(f"  干旱指数 = {PET/P:.2f}")
    
    print(f"\n弹性系数:")
    print(f"  εP = {elasticities['epsilon_P']:.3f}")
    print(f"    → 降水增加1%，径流增加{elasticities['epsilon_P']:.3f}%")
    
    print(f"  εPET = {elasticities['epsilon_PET']:.3f}")
    print(f"    → PET增加1%，径流减少{abs(elasticities['epsilon_PET']):.3f}%")
    
    print(f"  εn = {elasticities['epsilon_n']:.3f}")
    print(f"    → 参数n增加1%，径流减少{abs(elasticities['epsilon_n']):.3f}%")
    
    # 敏感性分析
    print(f"\n敏感性排序:")
    sensitivities = {
        '降水(P)': abs(elasticities['epsilon_P']),
        'PET': abs(elasticities['epsilon_PET']),
        '下垫面(n)': abs(elasticities['epsilon_n'])
    }
    ranked = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for i, (factor, value) in enumerate(ranked, 1):
        print(f"  {i}. {factor}: {value:.3f}")


def example_4_attribution_analysis():
    """
    示例4: 完整的归因分析
    
    分析两个时期径流变化的原因（main.tex Step 1-5）
    """
    print("\n\n" + "=" * 60)
    print("示例4: 径流变化归因分析")
    print("=" * 60)
    
    model = BudykoModel()
    
    # 假设某流域两个时期的数据
    # 基准期（1960-1985）
    P_base = 850
    PET_base = 1100
    Q_n_base = 260
    
    # 影响期（1986-2016）
    P_impact = 800
    PET_impact = 1200
    Q_n_impact = 190
    
    print(f"\n基准期（1960-1985）:")
    print(f"  P = {P_base} mm/year")
    print(f"  PET = {PET_base} mm/year")
    print(f"  Q_n = {Q_n_base} mm/year")
    
    print(f"\n影响期（1986-2016）:")
    print(f"  P = {P_impact} mm/year")
    print(f"  PET = {PET_impact} mm/year")
    print(f"  Q_n = {Q_n_impact} mm/year")
    
    # 执行归因分析
    print("\n正在进行归因分析...")
    attribution = model.calculate_runoff_change_attribution(
        P_base, PET_base, Q_n_base,
        P_impact, PET_impact, Q_n_impact
    )
    
    # 输出结果
    print(f"\n" + "-" * 60)
    print("归因分析结果")
    print("-" * 60)
    
    print(f"\n1. 校准参数:")
    print(f"   全时段 n = {attribution['n_overall']:.3f}")
    print(f"   基准期 n = {attribution['n_base']:.3f}")
    print(f"   影响期 n = {attribution['n_impact']:.3f}")
    print(f"   Δn = {attribution['delta_n']:.3f}")
    
    print(f"\n2. 弹性系数:")
    print(f"   εP = {attribution['epsilon_P']:.3f}")
    print(f"   εPET = {attribution['epsilon_PET']:.3f}")
    print(f"   εn = {attribution['epsilon_n']:.3f}")
    
    print(f"\n3. 气候变化:")
    print(f"   ΔP = {attribution['delta_P']:.2f} mm/year")
    print(f"   ΔPET = {attribution['delta_PET']:.2f} mm/year")
    
    print(f"\n4. 径流变化分解:")
    print(f"   观测径流变化 ΔQ_n = {attribution['delta_Q_n']:.2f} mm/year")
    print(f"   ├─ 气候贡献 ΔQ_n,CCV = {attribution['delta_Q_n_CCV']:.2f} mm/year")
    print(f"   └─ LUCC贡献 ΔQ_n,LUCC = {attribution['delta_Q_n_LUCC']:.2f} mm/year")
    print(f"   模拟总变化 = {attribution['delta_Q_n_simulated']:.2f} mm/year")
    
    # 计算贡献率
    delta_Q = attribution['delta_Q_n']
    if abs(delta_Q) > 1:  # 变化显著
        C_CCV = attribution['delta_Q_n_CCV'] / delta_Q * 100
        C_LUCC = attribution['delta_Q_n_LUCC'] / delta_Q * 100
        
        print(f"\n5. 贡献率:")
        print(f"   气候变化 (CCV) = {C_CCV:.1f}%")
        print(f"   土地利用 (LUCC) = {C_LUCC:.1f}%")


def example_5_climate_scenarios():
    """
    示例5: 不同气候情景下的Budyko行为
    
    对比湿润、半干旱和干旱三种气候类型
    """
    print("\n\n" + "=" * 60)
    print("示例5: 不同气候情景对比")
    print("=" * 60)
    
    model = BudykoModel()
    
    # 定义三种气候情景
    scenarios = {
        '湿润区': {'P': 1500, 'PET': 800, 'n': 2.8},
        '半干旱区': {'P': 600, 'PET': 1000, 'n': 2.0},
        '干旱区': {'P': 250, 'PET': 1500, 'n': 1.5}
    }
    
    print(f"\n{'气候类型':<10} {'P(mm)':<8} {'PET(mm)':<9} {'φ':<6} {'n':<6} {'E(mm)':<8} {'Q_n(mm)':<9} {'Q_n/P':<8}")
    print("-" * 80)
    
    for climate, params in scenarios.items():
        P = params['P']
        PET = params['PET']
        n = params['n']
        
        phi = PET / P
        E = model.calculate_actual_ET(P, PET, n)
        Q_n = model.calculate_naturalized_runoff(P, PET, n)
        runoff_ratio = Q_n / P
        
        print(f"{climate:<10} {P:<8} {PET:<9} {phi:<6.2f} {n:<6.1f} {E:<8.1f} {Q_n:<9.1f} {runoff_ratio:<8.2f}")
    
    print(f"\n观察:")
    print(f"  1. 湿润区: φ < 1, E受能量限制（E ≈ PET），产流率高")
    print(f"  2. 半干旱区: φ ≈ 1-2, E介于水分和能量限制之间")
    print(f"  3. 干旱区: φ > 2, E受水分限制（E ≈ P），产流率极低")


def example_6_sensitivity_to_n():
    """
    示例6: 径流对参数n的敏感性
    
    固定P和PET，改变n值，观察径流变化
    """
    print("\n\n" + "=" * 60)
    print("示例6: 参数n敏感性分析")
    print("=" * 60)
    
    model = BudykoModel()
    
    # 固定气候条件
    P = 800
    PET = 1200
    
    # 测试不同的n值
    n_values = np.arange(1.0, 4.1, 0.5)
    Q_n_values = []
    
    print(f"\n固定条件: P = {P} mm/year, PET = {PET} mm/year")
    print(f"\n{'n':<6} {'Q_n (mm)':<12} {'Q_n/P':<10} {'流域特征'}")
    print("-" * 50)
    
    for n in n_values:
        Q_n = model.calculate_naturalized_runoff(P, PET, n)
        Q_n_values.append(Q_n)
        runoff_ratio = Q_n / P
        
        # 根据径流系数判断流域特征
        if runoff_ratio > 0.35:
            feature = "高产流（如森林流域）"
        elif runoff_ratio > 0.20:
            feature = "中等产流"
        else:
            feature = "低产流（如草地/裸地）"
        
        print(f"{n:<6.1f} {Q_n:<12.2f} {runoff_ratio:<10.2f} {feature}")
    
    print(f"\n结论: n值增加，径流减少（蒸散发能力增强）")
    print(f"  Δn = +3.0 时，径流减少约{(Q_n_values[0] - Q_n_values[-1]):.0f} mm/year")


def example_7_integration_with_previous_modules():
    """
    示例7: 与前置模块集成
    
    模拟完整的数据处理流程：
    GRDC数据 → 气候数据 → PET计算 → Budyko归因
    """
    print("\n\n" + "=" * 60)
    print("示例7: 模块集成示范")
    print("=" * 60)
    
    print("\n完整数据处理流程:")
    print("  1. 模块1 (GRDCParser): 解析径流观测数据")
    print("  2. 模块2 (ClimateDataProcessor): 处理ISIMIP气候数据")
    print("  3. 模块3 (PETCalculator): 计算潜在蒸散发")
    print("  4. 模块4 (BudykoModel): 归因分析 ← 当前模块")
    
    print("\n模拟数据流（某真实流域）:")
    
    # 步骤1: 来自GRDC（模块1输出）
    print("\n[模块1输出] GRDC观测径流:")
    Q_o_1960_1985 = pd.Series([245, 250, 260, 255, 248], name='Q_o_mm')
    Q_o_1986_2016 = pd.Series([185, 190, 195, 180, 188], name='Q_o_mm')
    print(f"  基准期平均: {Q_o_1960_1985.mean():.2f} mm/year")
    print(f"  影响期平均: {Q_o_1986_2016.mean():.2f} mm/year")
    
    # 步骤2: 来自ISIMIP（模块2输出）
    print("\n[模块2输出] 气候变量:")
    P_1960_1985 = pd.Series([845, 850, 860, 840, 855], name='P_mm')
    P_1986_2016 = pd.Series([795, 800, 810, 790, 805], name='P_mm')
    print(f"  基准期降水: {P_1960_1985.mean():.2f} mm/year")
    print(f"  影响期降水: {P_1986_2016.mean():.2f} mm/year")
    
    # 步骤3: PET计算（模块3输出）
    print("\n[模块3输出] PET计算:")
    PET_1960_1985 = pd.Series([1095, 1100, 1110, 1090, 1105], name='PET_mm')
    PET_1986_2016 = pd.Series([1195, 1200, 1210, 1190, 1205], name='PET_mm')
    print(f"  基准期PET: {PET_1960_1985.mean():.2f} mm/year")
    print(f"  影响期PET: {PET_1986_2016.mean():.2f} mm/year")
    
    # 步骤4: Budyko归因（当前模块）
    print("\n[模块4执行] Budyko归因分析:")
    
    model = BudykoModel()
    
    # 假设已经进行了用水还原（需要Huang et al.数据）
    # Q_n = Q_o + W_consumption
    Q_n_1960_1985 = Q_o_1960_1985.mean()  # 简化：假设基准期无取用水
    Q_n_1986_2016 = Q_o_1986_2016.mean() + 15  # 加上估算的耗水量
    
    attribution = model.calculate_runoff_change_attribution(
        P_1960_1985.mean(), PET_1960_1985.mean(), Q_n_1960_1985,
        P_1986_2016.mean(), PET_1986_2016.mean(), Q_n_1986_2016
    )
    
    print(f"\n归因结果:")
    print(f"  ΔQ_o = {Q_o_1986_2016.mean() - Q_o_1960_1985.mean():.2f} mm/year")
    print(f"  ├─ 气候贡献: {attribution['delta_Q_n_CCV']:.2f} mm/year")
    print(f"  ├─ LUCC贡献: {attribution['delta_Q_n_LUCC']:.2f} mm/year")
    print(f"  └─ WADR贡献: {(Q_o_1986_2016.mean() - Q_n_1986_2016) - (Q_o_1960_1985.mean() - Q_n_1960_1985):.2f} mm/year")
    
    print(f"\n→ 下一步: 将结果输入模块7-8进行ISIMIP模型归因")


# 主执行函数
def main():
    """运行所有示例"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Budyko核心方程模块使用示例" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    examples = [
        example_1_basic_budyko_calculation,
        example_2_parameter_calibration,
        example_3_elasticity_analysis,
        example_4_attribution_analysis,
        example_5_climate_scenarios,
        example_6_sensitivity_to_n,
        example_7_integration_with_previous_modules
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n错误: {str(e)}")
            continue
    
    print("\n\n" + "=" * 60)
    print("所有示例运行完毕")
    print("=" * 60)
    print("\n提示:")
    print("  - 修改示例中的参数值，观察Budyko方程的行为")
    print("  - 使用真实的GRDC/ISIMIP数据替换示例数据")
    print("  - 结合可视化工具绘制Budyko曲线")


if __name__ == "__main__":
    main()
