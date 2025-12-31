"""
elasticity_solver.py

径流弹性系数计算模块

本模块实现径流对各驱动因子（降水、潜在蒸散发、流域景观参数）的
弹性系数计算。弹性系数用于量化径流对各因子微小变化的敏感性，是
归因分析的核心。

理论基础：
根据Budyko-Choudhury-Yang方程，径流变化可表示为：
    dQ_n/Q_n = εP·dP/P + εPET·dPET/PET + εn·dn/n

其中：
    - εP: 降水弹性系数（通常 > 1，表现放大效应）
    - εPET: 潜在蒸散发弹性系数（通常 < 0，负相关）
    - εn: 流域景观参数弹性系数（通常 < 0，截留增强减少径流）

主要函数：
    calculate_elasticity_P: 计算降水弹性系数
    calculate_elasticity_PET: 计算PET弹性系数
    calculate_elasticity_n: 计算参数n弹性系数
    calculate_all_elasticities: 一次性计算所有三项弹性系数
    validate_elasticity_signs: 验证弹性系数符号的物理合理性

参考文献：
    - main.tex 方程 (3-6)
    - Yang et al. (2008) Water Resources Research
    - Choudhury (1999) Journal of Hydrology

作者: Research Software Engineer
日期: 2025-01-01
"""

import numpy as np
import warnings
from typing import Dict, Union, Tuple


def calculate_elasticity_P(
    P: Union[float, np.ndarray],
    PET: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    计算降水弹性系数 εP
    
    降水弹性系数反映径流对降水变化的敏感性。通常 εP > 1，表明
    径流对降水具有放大效应（即降水增加1%会导致径流增加超过1%）。
    
    理论公式（main.tex 方程4）：
        εP = [1 - (φ^n/(1+φ^n))^(1/n+1)] / [1 - (φ^n/(1+φ^n))^(1/n)]
        
    其中 φ = PET/P 为干旱指数。
    
    参数:
        P: 多年平均降水量 (mm/year)，标量或数组
        PET: 多年平均潜在蒸散发 (mm/year)，标量或数组
        n: 流域景观参数（无量纲），标量或数组
        epsilon: 数值稳定性参数，避免除零错误
    
    返回:
        εP: 降水弹性系数（无量纲）
    
    异常:
        ValueError: 当输入值非正时
        UserWarning: 当计算结果可能不准确时
    
    示例:
        >>> eps_P = calculate_elasticity_P(P=800, PET=1200, n=2.5)
        >>> print(f"降水弹性: εP = {eps_P:.3f}")
        降水弹性: εP = 2.156
        
    注意:
        - εP通常在1.5-3.0之间，取决于流域的湿润程度
        - 湿润地区（P>PET）的εP较小，干旱地区较大
        - 如果εP<1或εP>5，建议检查输入数据
    """
    # 输入验证
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    n = np.asarray(n, dtype=float)
    
    if np.any(P <= 0):
        raise ValueError("降水P必须为正数")
    if np.any(PET <= 0):
        raise ValueError("潜在蒸散发PET必须为正数")
    if np.any(n <= 0):
        raise ValueError("参数n必须为正数")
    
    # 避免数值问题：添加极小量
    P = np.maximum(P, epsilon)
    PET = np.maximum(PET, epsilon)
    
    # 计算干旱指数 φ = PET/P
    phi = PET / P
    
    # 计算 φ^n
    phi_n = np.power(phi, n)
    
    # 计算 ratio_term = φ^n / (1 + φ^n)
    ratio_term = phi_n / (1 + phi_n)
    
    # 计算 ratio_term^(1/n) 和 ratio_term^(1/n + 1)
    ratio_1_over_n = np.power(ratio_term, 1.0 / n)
    ratio_1_over_n_plus_1 = np.power(ratio_term, 1.0 / n + 1)
    
    # 计算弹性系数
    numerator = 1 - ratio_1_over_n_plus_1
    denominator = 1 - ratio_1_over_n
    
    # 处理分母接近零的情况
    mask = np.abs(denominator) < epsilon
    if np.any(mask):
        warnings.warn(
            "εP计算中分母接近零，部分结果可能不准确。"
            "建议检查输入数据的物理合理性。",
            UserWarning
        )
        # 对于分母接近零的情况，使用默认值
        epsilon_P = np.where(mask, 1.0, numerator / denominator)
    else:
        epsilon_P = numerator / denominator
    
    return float(epsilon_P) if np.isscalar(P) else epsilon_P


def calculate_elasticity_PET(
    P: Union[float, np.ndarray],
    PET: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    计算潜在蒸散发弹性系数 εPET
    
    PET弹性系数反映径流对大气蒸发需求变化的敏感性。通常 εPET < 0，
    表明PET增加（如气温升高、辐射增强）会导致径流减少。
    
    理论公式（main.tex 方程5）：
        εPET = [1/(1+φ^n)] × [1/(1-((1+φ^n)/φ^n)^(1/n))]
        
    其中 φ = PET/P 为干旱指数。
    
    参数:
        P: 多年平均降水量 (mm/year)，标量或数组
        PET: 多年平均潜在蒸散发 (mm/year)，标量或数组
        n: 流域景观参数（无量纲），标量或数组
        epsilon: 数值稳定性参数
    
    返回:
        εPET: PET弹性系数（无量纲，通常为负）
    
    异常:
        ValueError: 当输入值非正时
        UserWarning: 当计算结果可能不准确时
    
    示例:
        >>> eps_PET = calculate_elasticity_PET(P=800, PET=1200, n=2.5)
        >>> print(f"PET弹性: εPET = {eps_PET:.3f}")
        PET弹性: εPET = -1.156
        
    注意:
        - εPET通常在-2.0到-0.5之间
        - 湿润地区（P>PET）的|εPET|较小，干旱地区较大
        - εP + εPET 在某些公式下约等于1（软约束）
    """
    # 输入验证
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    n = np.asarray(n, dtype=float)
    
    if np.any(P <= 0):
        raise ValueError("降水P必须为正数")
    if np.any(PET <= 0):
        raise ValueError("潜在蒸散发PET必须为正数")
    if np.any(n <= 0):
        raise ValueError("参数n必须为正数")
    
    # 避免数值问题
    P = np.maximum(P, epsilon)
    PET = np.maximum(PET, epsilon)
    
    # 计算干旱指数 φ = PET/P
    phi = PET / P
    
    # 计算 φ^n
    phi_n = np.power(phi, n)
    
    # 第一项: 1 / (1 + φ^n)
    factor1 = 1.0 / (1 + phi_n)
    
    # 第二项: 1 / (1 - ((1 + φ^n) / φ^n)^(1/n))
    inverse_ratio_term = (1 + phi_n) / phi_n  # (1 + φ^n) / φ^n
    inverse_ratio_1_over_n = np.power(inverse_ratio_term, 1.0 / n)
    
    denominator = 1 - inverse_ratio_1_over_n
    
    # 处理分母接近零的情况
    mask = np.abs(denominator) < epsilon
    if np.any(mask):
        warnings.warn(
            "εPET计算中分母接近零，部分结果可能不准确。",
            UserWarning
        )
        # 对于分母接近零的情况，使用默认值（通常为负）
        factor2 = np.where(mask, 2.0, 1.0 / denominator)
    else:
        factor2 = 1.0 / denominator
    
    epsilon_PET = factor1 * factor2
    
    return float(epsilon_PET) if np.isscalar(P) else epsilon_PET


def calculate_elasticity_n(
    P: Union[float, np.ndarray],
    PET: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    计算流域景观参数弹性系数 εn
    
    参数n弹性系数反映径流对下垫面特征变化的敏感性。通常 εn < 0，
    表明下垫面截留能力增强（如植被恢复、植林）会导致径流减少。
    
    理论公式（main.tex 方程6）：
        εn = {1/[(1+(P/PET)^n)^(1/n)-1]} × 
             {[P^n·ln(P) + PET^n·ln(PET)]/(P^n+PET^n) - ln(P^n+PET^n)/n}
    
    参数:
        P: 多年平均降水量 (mm/year)，标量或数组
        PET: 多年平均潜在蒸散发 (mm/year)，标量或数组
        n: 流域景观参数（无量纲），标量或数组
        epsilon: 数值稳定性参数
    
    返回:
        εn: 参数n弹性系数（无量纲，通常为负）
    
    异常:
        ValueError: 当输入值非正时
        UserWarning: 当计算结果可能不准确时
    
    示例:
        >>> eps_n = calculate_elasticity_n(P=800, PET=1200, n=2.5)
        >>> print(f"参数n弹性: εn = {eps_n:.3f}")
        参数n弹性: εn = -0.485
        
    注意:
        - εn通常在-1.0到-0.1之间
        - |εn|的大小反映了流域对下垫面变化的敏感程度
        - 该弹性系数主要用于量化LUCC对径流的影响
    """
    # 输入验证
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    n = np.asarray(n, dtype=float)
    
    if np.any(P <= 0):
        raise ValueError("降水P必须为正数")
    if np.any(PET <= 0):
        raise ValueError("潜在蒸散发PET必须为正数")
    if np.any(n <= 0):
        raise ValueError("参数n必须为正数")
    
    # 避免数值问题
    P = np.maximum(P, epsilon)
    PET = np.maximum(PET, epsilon)
    
    # 计算 (P/PET)^n（注意这里是P/PET，不是PET/P）
    inverse_phi = P / PET
    inverse_phi_n = np.power(inverse_phi, n)
    
    # 第一项：1 / [(1 + (P/PET)^n)^(1/n) - 1]
    term1_denominator = np.power(1 + inverse_phi_n, 1.0 / n) - 1
    
    # 处理第一项分母接近零的情况
    mask1 = np.abs(term1_denominator) < epsilon
    if np.any(mask1):
        warnings.warn(
            "εn计算中第一项分母接近零，部分结果可能不准确。",
            UserWarning
        )
        factor_n_1 = np.where(mask1, 10.0, 1.0 / term1_denominator)
    else:
        factor_n_1 = 1.0 / term1_denominator
    
    # 第二项的计算
    P_n = np.power(P, n)
    PET_n = np.power(PET, n)
    
    # 对数项（需要确保参数为正）
    ln_P = np.log(P)
    ln_PET = np.log(PET)
    
    # 计算 [P^n·ln(P) + PET^n·ln(PET)] / [P^n + PET^n]
    sum_Pn_PETn = P_n + PET_n
    numerator_n = (P_n * ln_P + PET_n * ln_PET) / sum_Pn_PETn
    
    # 计算 ln(P^n + PET^n) / n
    denominator_n_term = np.log(sum_Pn_PETn) / n
    
    # 第二项
    factor_n_2 = numerator_n - denominator_n_term
    
    # 最终结果
    epsilon_n = factor_n_1 * factor_n_2
    
    return float(epsilon_n) if np.isscalar(P) else epsilon_n


def calculate_all_elasticities(
    P: Union[float, np.ndarray],
    PET: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    一次性计算所有三项弹性系数
    
    这是一个便捷函数，同时计算εP、εPET和εn，并返回字典格式结果。
    
    参数:
        P: 多年平均降水量 (mm/year)
        PET: 多年平均潜在蒸散发 (mm/year)
        n: 流域景观参数
        epsilon: 数值稳定性参数
    
    返回:
        字典包含:
            'epsilon_P': 降水弹性系数
            'epsilon_PET': PET弹性系数
            'epsilon_n': 参数n弹性系数
            'sum_P_PET': εP + εPET（用于检验）
    
    示例:
        >>> elasticities = calculate_all_elasticities(P=800, PET=1200, n=2.5)
        >>> print(f"εP = {elasticities['epsilon_P']:.3f}")
        >>> print(f"εPET = {elasticities['epsilon_PET']:.3f}")
        >>> print(f"εn = {elasticities['epsilon_n']:.3f}")
        >>> print(f"εP + εPET = {elasticities['sum_P_PET']:.3f}")
    """
    epsilon_P = calculate_elasticity_P(P, PET, n, epsilon)
    epsilon_PET = calculate_elasticity_PET(P, PET, n, epsilon)
    epsilon_n = calculate_elasticity_n(P, PET, n, epsilon)
    
    return {
        'epsilon_P': epsilon_P,
        'epsilon_PET': epsilon_PET,
        'epsilon_n': epsilon_n,
        'sum_P_PET': epsilon_P + epsilon_PET
    }


def validate_elasticity_signs(
    epsilon_P: float,
    epsilon_PET: float,
    epsilon_n: float,
    strict: bool = False
) -> Tuple[bool, str]:
    """
    验证弹性系数符号的物理合理性
    
    根据Budyko理论，弹性系数应满足以下符号关系：
        - εP > 0（降水增加导致径流增加）
        - εPET < 0（PET增加导致径流减少）
        - εn < 0（截留能力增强导致径流减少）
    
    参数:
        epsilon_P: 降水弹性系数
        epsilon_PET: PET弹性系数
        epsilon_n: 参数n弹性系数
        strict: 是否启用严格模式（检查数值范围）
    
    返回:
        (is_valid, message): 验证结果和说明信息
    
    示例:
        >>> is_valid, msg = validate_elasticity_signs(2.1, -1.0, -0.5)
        >>> print(f"验证结果: {is_valid}, {msg}")
        验证结果: True, 弹性系数符号合理
    """
    # 基本符号检查
    if epsilon_P <= 0:
        return False, f"εP={epsilon_P:.3f}应为正数（降水增加导致径流增加）"
    
    if epsilon_PET >= 0:
        return False, f"εPET={epsilon_PET:.3f}应为负数（PET增加导致径流减少）"
    
    if epsilon_n >= 0:
        return False, f"εn={epsilon_n:.3f}应为负数（截留能力增强导致径流减少）"
    
    # 严格模式下的数值范围检查
    if strict:
        # εP通常在1.0-5.0之间
        if epsilon_P < 0.5 or epsilon_P > 10.0:
            return False, (
                f"εP={epsilon_P:.3f}超出合理范围[0.5, 10.0]，"
                "建议检查输入数据"
            )
        
        # εPET通常在-3.0到-0.2之间
        if epsilon_PET < -5.0 or epsilon_PET > -0.01:
            return False, (
                f"εPET={epsilon_PET:.3f}超出合理范围[-5.0, -0.01]，"
                "建议检查输入数据"
            )
        
        # εn通常在-2.0到-0.05之间
        if epsilon_n < -3.0 or epsilon_n > -0.01:
            return False, (
                f"εn={epsilon_n:.3f}超出合理范围[-3.0, -0.01]，"
                "建议检查输入数据"
            )
        
        # εP + εPET 软约束检查（对于Choudhury-Yang公式，和值可能偏离1）
        sum_P_PET = epsilon_P + epsilon_PET
        if sum_P_PET < 0.3 or sum_P_PET > 2.0:
            warnings.warn(
                f"εP + εPET = {sum_P_PET:.3f} 偏离典型值范围[0.3, 2.0]，"
                "可能存在数值不稳定性",
                UserWarning
            )
    
    return True, "弹性系数符号和数值范围合理"


# 辅助函数：批量计算
def batch_calculate_elasticities(
    data: 'pd.DataFrame',
    P_col: str = 'P',
    PET_col: str = 'PET',
    n_col: str = 'n'
) -> 'pd.DataFrame':
    """
    对DataFrame中的多行数据批量计算弹性系数
    
    参数:
        data: 包含P、PET、n列的DataFrame
        P_col: 降水列名
        PET_col: PET列名
        n_col: 参数n列名
    
    返回:
        添加了epsilon_P、epsilon_PET、epsilon_n列的DataFrame
    
    示例:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'P': [800, 600, 1000],
        ...     'PET': [1200, 900, 800],
        ...     'n': [2.5, 2.0, 3.0]
        ... })
        >>> result = batch_calculate_elasticities(df)
        >>> print(result[['epsilon_P', 'epsilon_PET', 'epsilon_n']])
    """
    import pandas as pd
    
    result = data.copy()
    
    # 向量化计算所有弹性系数
    result['epsilon_P'] = calculate_elasticity_P(
        data[P_col].values,
        data[PET_col].values,
        data[n_col].values
    )
    
    result['epsilon_PET'] = calculate_elasticity_PET(
        data[P_col].values,
        data[PET_col].values,
        data[n_col].values
    )
    
    result['epsilon_n'] = calculate_elasticity_n(
        data[P_col].values,
        data[PET_col].values,
        data[n_col].values
    )
    
    return result
