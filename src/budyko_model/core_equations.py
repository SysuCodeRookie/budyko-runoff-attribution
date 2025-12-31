"""
Budyko核心方程模块

实现Budyko-Choudhury-Yang水热平衡方程，包括：
1. 实际蒸散发计算（E）
2. 天然径流计算（Q_n）
3. 参数n的反演（校准）
4. 径流弹性系数计算（εP, εPET, εn）

基于main.tex中的理论框架和土木工程方法指南的算法规范。

作者: Budyko归因分析系统开发团队
日期: 2025-01-01
"""

import numpy as np
import pandas as pd
from scipy import optimize
import warnings
from typing import Union, Tuple, Dict


# ============================================================================
# 数值稳定性工具函数（根据"代码撰写总体建议"第六章要求）
# ============================================================================

def safe_divide(numerator: np.ndarray, 
                denominator: np.ndarray, 
                fill_value: float = np.nan) -> np.ndarray:
    """
    安全除法，避免除零错误
    
    Parameters
    ----------
    numerator : array_like
        分子
    denominator : array_like
        分母
    fill_value : float, optional
        分母为0时的填充值，默认为NaN
        
    Returns
    -------
    result : ndarray
        除法结果，分母为0的位置用fill_value填充
        
    Examples
    --------
    >>> safe_divide(np.array([1, 2, 3]), np.array([2, 0, 4]))
    array([0.5, nan, 0.75])
    """
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = fill_value
    
    return result


def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    安全对数运算，确保输入 > 0
    
    Parameters
    ----------
    x : array_like
        输入值
    epsilon : float, optional
        最小值平滑参数，默认1e-10
        
    Returns
    -------
    result : ndarray
        对数结果
        
    Notes
    -----
    用于弹性系数εn的计算，避免ln(0)错误。
    对于极小值，使用ln(epsilon)代替。
    
    Examples
    --------
    >>> safe_log(np.array([1, 0.1, 0, -1]))
    array([ 0.        , -2.30258509, -23.02585093,        nan])
    """
    x = np.asarray(x)
    x_safe = np.maximum(x, epsilon)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(x_safe)
        # 对于负数，仍然返回NaN
        result[x < 0] = np.nan
    
    return result


def safe_power(base: np.ndarray, exponent: float, 
               epsilon: float = 1e-10) -> np.ndarray:
    """
    安全幂运算，避免负数的非整数次幂
    
    Parameters
    ----------
    base : array_like
        底数
    exponent : float
        指数
    epsilon : float, optional
        最小正值，默认1e-10
        
    Returns
    -------
    result : ndarray
        幂运算结果
        
    Notes
    -----
    Budyko方程中需要计算P^n和PET^n，其中n可能是非整数。
    如果P或PET为负值或零，会导致复数域错误。
    
    Examples
    --------
    >>> safe_power(np.array([2, 0, -1]), 2.5)
    array([5.65685425e+00, 3.16227766e-25,            nan])
    """
    base = np.asarray(base)
    base_safe = np.maximum(base, epsilon)
    
    with np.errstate(invalid='ignore'):
        result = np.power(base_safe, exponent)
        # 对于原始负值，返回NaN
        result[base < 0] = np.nan
    
    return result


class BudykoModel:
    """
    Budyko水热平衡模型核心类
    
    实现Choudhury-Yang参数化Budyko方程，用于：
    - 计算实际蒸散发（E）和天然径流（Q_n）
    - 反演流域景观参数（n）
    - 计算径流对气候和下垫面变化的弹性系数
    
    理论基础：
    E = (P × PET) / (P^n + PET^n)^(1/n)
    Q_n = P - E
    
    参数:
        epsilon (float): 数值计算的极小值平滑参数，避免除零和对数错误
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        初始化Budyko模型
        
        参数:
            epsilon: 极小值平滑参数，默认1e-10
        """
        self.epsilon = epsilon
    
    def calculate_actual_ET(
        self, 
        P: Union[float, np.ndarray], 
        PET: Union[float, np.ndarray], 
        n: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        计算实际蒸散发（E）- Choudhury-Yang方程
        
        公式: E = (P × PET) / (P^n + PET^n)^(1/n)
        
        参数:
            P: 降水量 (mm/year)，标量或数组
            PET: 潜在蒸散发 (mm/year)，标量或数组
            n: 流域景观参数（无量纲），标量或数组
        
        返回:
            E: 实际蒸散发 (mm/year)
        
        异常:
            ValueError: 当输入值非法时（负值、P和PET形状不匹配等）
        
        示例:
            >>> model = BudykoModel()
            >>> E = model.calculate_actual_ET(P=800, PET=1200, n=2.5)
            >>> print(f"实际蒸散发: {E:.2f} mm/year")
        """
        # 输入验证
        P = np.asarray(P, dtype=float)
        PET = np.asarray(PET, dtype=float)
        n = np.asarray(n, dtype=float)
        
        if np.any(P < 0):
            raise ValueError("降水量P不能为负值")
        if np.any(PET < 0):
            raise ValueError("潜在蒸散发PET不能为负值")
        if np.any(n <= 0):
            raise ValueError("参数n必须为正数")
        
        # 避免除零：为零值添加极小量
        P = np.maximum(P, self.epsilon)
        PET = np.maximum(PET, self.epsilon)
        
        # Choudhury-Yang方程
        # E = (P × PET) / (P^n + PET^n)^(1/n)
        denominator = np.power(np.power(P, n) + np.power(PET, n), 1.0 / n)
        E = (P * PET) / denominator
        
        return E
    
    def calculate_naturalized_runoff(
        self, 
        P: Union[float, np.ndarray], 
        PET: Union[float, np.ndarray], 
        n: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        计算天然径流（Q_n）- 基于水量平衡
        
        公式: Q_n = P - E = P - (P × PET) / (P^n + PET^n)^(1/n)
        
        参数:
            P: 降水量 (mm/year)
            PET: 潜在蒸散发 (mm/year)
            n: 流域景观参数（无量纲）
        
        返回:
            Q_n: 天然径流 (mm/year)
        
        注意:
            Q_n应满足物理约束: 0 <= Q_n <= P
            若违反约束会发出警告
        
        示例:
            >>> model = BudykoModel()
            >>> Q_n = model.calculate_naturalized_runoff(P=800, PET=1200, n=2.5)
            >>> print(f"天然径流: {Q_n:.2f} mm/year")
        """
        E = self.calculate_actual_ET(P, PET, n)
        Q_n = P - E
        
        # 物理一致性检查
        if np.any(Q_n < -self.epsilon):
            warnings.warn(
                f"计算得到负径流值: Q_n={Q_n.min():.2f} mm/year，"
                "可能是输入数据异常或参数n不合理",
                UserWarning
            )
        
        if np.any(Q_n > P + self.epsilon):
            warnings.warn(
                f"径流超过降水量: Q_n={Q_n.max():.2f} > P={np.max(P):.2f}",
                UserWarning
            )
        
        # 将负值截断为0
        Q_n = np.maximum(Q_n, 0)
        
        return Q_n
    
    def calibrate_parameter_n(
        self,
        P: float,
        PET: float,
        Q_n: float,
        n_bounds: Tuple[float, float] = (0.1, 10.0),
        method: str = 'brentq'
    ) -> float:
        """
        反演流域景观参数n（模型校准）
        
        已知观测的P, PET, Q_n，求解使得模型输出Q_n与观测值一致的参数n。
        这是一个根查找问题：f(n) = Q_n_simulated(P, PET, n) - Q_n_observed = 0
        
        参数:
            P: 多年平均降水量 (mm/year)
            PET: 多年平均潜在蒸散发 (mm/year)
            Q_n: 多年平均观测天然径流 (mm/year)
            n_bounds: 参数n的搜索范围，默认(0.1, 10.0)
            method: 求解方法，'brentq'（Brent方法）或'newton'（牛顿法）
        
        返回:
            n: 校准得到的流域景观参数
        
        异常:
            ValueError: 当水量平衡违背（Q_n >= P）或无法找到解时
        
        示例:
            >>> model = BudykoModel()
            >>> n = model.calibrate_parameter_n(P=800, PET=1200, Q_n=200)
            >>> print(f"校准参数: n = {n:.3f}")
        
        注意:
            - 必须满足物理约束: 0 < Q_n < P
            - 若P和PET比值极端，可能无解（返回边界值）
            - 建议使用长序列（>10年）的平均值进行校准
        """
        # 输入验证
        if P <= 0 or PET <= 0:
            raise ValueError("P和PET必须为正数")
        
        if Q_n < 0:
            raise ValueError("天然径流Q_n不能为负值")
        
        # 关键物理约束：Q_n必须小于P
        if Q_n >= P - self.epsilon:
            raise ValueError(
                f"违反水量平衡: Q_n ({Q_n:.2f}) >= P ({P:.2f})，"
                "说明数据异常或存在跨流域调水"
            )
        
        # 定义目标函数：Q_n_simulated - Q_n_observed = 0
        def objective(n_trial):
            Q_n_sim = self.calculate_naturalized_runoff(P, PET, n_trial)
            return Q_n_sim - Q_n
        
        # 检查边界处的函数值
        try:
            f_lower = objective(n_bounds[0])
            f_upper = objective(n_bounds[1])
        except Exception as e:
            raise ValueError(f"计算边界值时出错: {str(e)}")
        
        # 如果边界处不变号，可能无解
        if f_lower * f_upper > 0:
            warnings.warn(
                f"在区间{n_bounds}内目标函数不变号，"
                f"f({n_bounds[0]})={f_lower:.2f}, f({n_bounds[1]})={f_upper:.2f}。"
                "尝试返回使|f(n)|最小的边界值",
                UserWarning
            )
            return n_bounds[0] if abs(f_lower) < abs(f_upper) else n_bounds[1]
        
        # 求解
        try:
            if method == 'brentq':
                # Brent方法（推荐）：结合二分法和插值，稳定且快速
                n_optimal = optimize.brentq(
                    objective, 
                    n_bounds[0], 
                    n_bounds[1],
                    xtol=1e-6,
                    maxiter=100
                )
            elif method == 'newton':
                # 牛顿法：需要导数，收敛更快但可能发散
                n_initial = (n_bounds[0] + n_bounds[1]) / 2
                n_optimal = optimize.newton(
                    objective,
                    n_initial,
                    maxiter=50
                )
                # 检查是否在范围内
                if not (n_bounds[0] <= n_optimal <= n_bounds[1]):
                    warnings.warn(
                        f"牛顿法结果n={n_optimal:.3f}超出范围，回退到Brent方法",
                        UserWarning
                    )
                    n_optimal = optimize.brentq(objective, n_bounds[0], n_bounds[1])
            else:
                raise ValueError(f"未知的求解方法: {method}")
            
            return float(n_optimal)
        
        except Exception as e:
            raise ValueError(f"参数n反演失败: {str(e)}")
    
    def calculate_elasticities(
        self,
        P: float,
        PET: float,
        n: float
    ) -> Dict[str, float]:
        """
        计算径流弹性系数（εP, εPET, εn）
        
        弹性系数定义为径流变化率与驱动因子变化率的比值，用于量化径流
        对降水、PET和下垫面参数的敏感性。
        
        公式（main.tex方程4-6）:
        - εP: 降水弹性，通常>1（放大效应）
        - εPET: PET弹性，通常<0（负相关）
        - εn: 参数n弹性，通常<0（截留能力增强减少径流）
        
        参数:
            P: 多年平均降水量 (mm/year)
            PET: 多年平均潜在蒸散发 (mm/year)
            n: 流域景观参数
        
        返回:
            字典包含:
                'epsilon_P': 降水弹性系数
                'epsilon_PET': PET弹性系数
                'epsilon_n': 参数n弹性系数
        
        示例:
            >>> model = BudykoModel()
            >>> elasticities = model.calculate_elasticities(P=800, PET=1200, n=2.5)
            >>> print(f"降水弹性: {elasticities['epsilon_P']:.3f}")
            >>> print(f"PET弹性: {elasticities['epsilon_PET']:.3f}")
        
        注意:
            - 公式包含对数运算，需要P和PET为正数
            - 建议使用长期平均值计算，避免年际噪声
        """
        # 输入验证
        if P <= 0 or PET <= 0 or n <= 0:
            raise ValueError("P, PET和n必须都为正数")
        
        # 避免数值问题
        P = max(P, self.epsilon)
        PET = max(PET, self.epsilon)
        
        # 定义干旱指数（中间变量，减少重复计算）
        phi = PET / P  # φ = PET/P
        
        # 计算常用中间项
        phi_n = np.power(phi, n)  # φ^n
        ratio_term = phi_n / (1 + phi_n)  # φ^n / (1 + φ^n)
        
        # 方程(4): εP - 降水弹性系数
        # εP = [1 - (ratio_term)^(1/n + 1)] / [1 - (ratio_term)^(1/n)]
        ratio_1_over_n = np.power(ratio_term, 1.0 / n)
        ratio_1_over_n_plus_1 = np.power(ratio_term, 1.0 / n + 1)
        
        numerator_P = 1 - ratio_1_over_n_plus_1
        denominator_P = 1 - ratio_1_over_n
        
        if abs(denominator_P) < self.epsilon:
            warnings.warn("εP计算中分母接近零，结果可能不准确", UserWarning)
            epsilon_P = 1.0  # 默认值
        else:
            epsilon_P = numerator_P / denominator_P
        
        # 方程(5): εPET - PET弹性系数
        # εPET = [1 / (1 + φ^n)] × [1 / (1 - [(1 + φ^n) / φ^n]^(1/n))]
        inverse_ratio_term = (1 + phi_n) / phi_n  # (1 + φ^n) / φ^n
        inverse_ratio_1_over_n = np.power(inverse_ratio_term, 1.0 / n)
        
        factor1 = 1.0 / (1 + phi_n)
        denominator_PET = 1 - inverse_ratio_1_over_n
        
        if abs(denominator_PET) < self.epsilon:
            warnings.warn("εPET计算中分母接近零，结果可能不准确", UserWarning)
            epsilon_PET = -0.5  # 默认值（通常为负）
        else:
            factor2 = 1.0 / denominator_PET
            epsilon_PET = factor1 * factor2
        
        # 方程(6): εn - 参数n弹性系数
        # εn = {1 / [(1 + (P/PET)^n)^(1/n) - 1]} × 
        #      {[P^n·ln(P) + PET^n·ln(PET)] / [P^n + PET^n] - ln(P^n + PET^n) / n}
        
        P_n = np.power(P, n)
        PET_n = np.power(PET, n)
        
        # 对数项（需要正值）
        ln_P = np.log(P)
        ln_PET = np.log(PET)
        
        # 第一项
        inverse_phi_n = np.power(1.0 / phi, n)  # (P/PET)^n
        term1_denominator = np.power(1 + inverse_phi_n, 1.0 / n) - 1
        
        if abs(term1_denominator) < self.epsilon:
            warnings.warn("εn计算中第一项分母接近零，结果可能不准确", UserWarning)
            epsilon_n = -0.1  # 默认值（通常为负小量）
        else:
            factor_n_1 = 1.0 / term1_denominator
            
            # 第二项
            sum_Pn_PETn = P_n + PET_n
            numerator_n = (P_n * ln_P + PET_n * ln_PET) / sum_Pn_PETn
            denominator_n_term = np.log(sum_Pn_PETn) / n
            factor_n_2 = numerator_n - denominator_n_term
            
            epsilon_n = factor_n_1 * factor_n_2
        
        return {
            'epsilon_P': float(epsilon_P),
            'epsilon_PET': float(epsilon_PET),
            'epsilon_n': float(epsilon_n)
        }
    
    def calculate_runoff_change_attribution(
        self,
        P_base: float,
        PET_base: float,
        Q_n_base: float,
        P_impact: float,
        PET_impact: float,
        Q_n_impact: float
    ) -> Dict[str, float]:
        """
        计算径流变化的归因分解（CCV和LUCC贡献）
        
        将观测的径流变化分解为：
        - ΔQ_n,CCV: 气候变化与变率导致的径流变化（通过ΔP和ΔPET）
        - ΔQ_n,LUCC: 土地利用/覆盖变化导致的径流变化（通过Δn）
        
        步骤（遵循main.tex Step 1-4）:
        1. 用全时段数据校准n
        2. 计算弹性系数（εP, εPET, εn）
        3. 分别校准两个时段的n值
        4. 应用方程(7)计算归因
        
        参数:
            P_base: 基准期平均降水 (mm/year)
            PET_base: 基准期平均PET (mm/year)
            Q_n_base: 基准期平均天然径流 (mm/year)
            P_impact: 影响期平均降水 (mm/year)
            PET_impact: 影响期平均PET (mm/year)
            Q_n_impact: 影响期平均天然径流 (mm/year)
        
        返回:
            字典包含:
                'n_overall': 全时段校准的n
                'n_base': 基准期n
                'n_impact': 影响期n
                'epsilon_P': 降水弹性
                'epsilon_PET': PET弹性
                'epsilon_n': 参数n弹性
                'delta_P': 降水变化量 (mm/year)
                'delta_PET': PET变化量 (mm/year)
                'delta_n': 参数n变化量
                'delta_Q_n': 天然径流变化量 (mm/year)
                'delta_Q_n_CCV': 气候变化贡献 (mm/year)
                'delta_Q_n_LUCC': 土地利用变化贡献 (mm/year)
                'delta_Q_n_simulated': Budyko模拟的总变化 (mm/year)
        
        示例:
            >>> model = BudykoModel()
            >>> attribution = model.calculate_runoff_change_attribution(
            ...     P_base=850, PET_base=1100, Q_n_base=250,
            ...     P_impact=800, PET_impact=1200, Q_n_impact=180
            ... )
            >>> print(f"气候变化贡献: {attribution['delta_Q_n_CCV']:.2f} mm/year")
            >>> print(f"土地利用贡献: {attribution['delta_Q_n_LUCC']:.2f} mm/year")
        """
        # Step 1: 用全时段平均值校准参数n（作为代表性n）
        P_overall = (P_base + P_impact) / 2
        PET_overall = (PET_base + PET_impact) / 2
        Q_n_overall = (Q_n_base + Q_n_impact) / 2
        
        try:
            n_overall = self.calibrate_parameter_n(P_overall, PET_overall, Q_n_overall)
        except ValueError as e:
            raise ValueError(f"全时段参数校准失败: {str(e)}")
        
        # Step 2: 计算弹性系数（使用全时段均值）
        elasticities = self.calculate_elasticities(P_overall, PET_overall, n_overall)
        epsilon_P = elasticities['epsilon_P']
        epsilon_PET = elasticities['epsilon_PET']
        epsilon_n = elasticities['epsilon_n']
        
        # Step 3: 分别校准两个时段的参数n
        try:
            n_base = self.calibrate_parameter_n(P_base, PET_base, Q_n_base)
        except ValueError as e:
            warnings.warn(f"基准期参数校准失败，使用全时段n值: {str(e)}", UserWarning)
            n_base = n_overall
        
        try:
            n_impact = self.calibrate_parameter_n(P_impact, PET_impact, Q_n_impact)
        except ValueError as e:
            warnings.warn(f"影响期参数校准失败，使用全时段n值: {str(e)}", UserWarning)
            n_impact = n_overall
        
        # 计算变化量
        delta_P = P_impact - P_base
        delta_PET = PET_impact - PET_base
        delta_n = n_impact - n_base
        delta_Q_n = Q_n_impact - Q_n_base
        
        # Step 4: 方程(7) - 归因分解
        # ΔQ̂_n = εP × (Q_n/P) × ΔP + εPET × (Q_n/PET) × ΔPET + εn × (Q_n/n) × Δn
        
        # 使用全时段均值作为归因基准
        delta_Q_n_CCV = (
            epsilon_P * (Q_n_overall / P_overall) * delta_P +
            epsilon_PET * (Q_n_overall / PET_overall) * delta_PET
        )
        
        delta_Q_n_LUCC = epsilon_n * (Q_n_overall / n_overall) * delta_n
        
        delta_Q_n_simulated = delta_Q_n_CCV + delta_Q_n_LUCC
        
        return {
            'n_overall': float(n_overall),
            'n_base': float(n_base),
            'n_impact': float(n_impact),
            'epsilon_P': float(epsilon_P),
            'epsilon_PET': float(epsilon_PET),
            'epsilon_n': float(epsilon_n),
            'delta_P': float(delta_P),
            'delta_PET': float(delta_PET),
            'delta_n': float(delta_n),
            'delta_Q_n': float(delta_Q_n),
            'delta_Q_n_CCV': float(delta_Q_n_CCV),
            'delta_Q_n_LUCC': float(delta_Q_n_LUCC),
            'delta_Q_n_simulated': float(delta_Q_n_simulated)
        }


# 辅助函数：物理一致性检查

def validate_water_balance(
    P: Union[float, np.ndarray],
    Q_n: Union[float, np.ndarray],
    threshold: float = 0.95
) -> Tuple[bool, str]:
    """
    检查水量平衡的物理一致性
    
    核心约束: Q_n < P（径流不能超过降水）
    
    参数:
        P: 降水量 (mm/year)
        Q_n: 天然径流 (mm/year)
        threshold: 警戒阈值，Q_n/P超过此值会发出警告（默认0.95）
    
    返回:
        (is_valid, message): 是否通过验证，以及详细信息
    
    示例:
        >>> is_valid, msg = validate_water_balance(P=800, Q_n=200)
        >>> if not is_valid:
        ...     print(f"验证失败: {msg}")
    """
    P = np.asarray(P)
    Q_n = np.asarray(Q_n)
    
    # 检查负值
    if np.any(P < 0):
        return False, "降水量P存在负值"
    if np.any(Q_n < 0):
        return False, "径流Q_n存在负值"
    
    # 检查Q_n是否超过P
    runoff_ratio = Q_n / (P + 1e-10)
    
    if np.any(runoff_ratio >= 1.0):
        max_ratio = np.max(runoff_ratio)
        return False, f"径流超过降水: 最大径流系数={max_ratio:.2f} (Q_n >= P)"
    
    if np.any(runoff_ratio > threshold):
        max_ratio = np.max(runoff_ratio)
        return True, f"径流系数偏高: 最大值={max_ratio:.2f} (接近P)，数据可能存在问题"
    
    return True, "水量平衡检查通过"


def calculate_aridity_index(
    PET: Union[float, np.ndarray],
    P: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    计算干旱指数（Aridity Index）
    
    定义: AI = PET / P
    
    分类（UNESCO标准）:
        < 0.05: 极端湿润
        0.05-0.20: 湿润
        0.20-0.50: 半湿润
        0.50-0.65: 半干旱
        > 0.65: 干旱/极端干旱
    
    参数:
        PET: 潜在蒸散发 (mm/year)
        P: 降水量 (mm/year)
    
    返回:
        aridity_index: 干旱指数（无量纲）
    
    示例:
        >>> AI = calculate_aridity_index(PET=1200, P=800)
        >>> print(f"干旱指数: {AI:.2f}")  # 输出: 1.50 (干旱)
    """
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    
    # 避免除零
    P = np.maximum(P, 1e-10)
    
    aridity_index = PET / P
    
    return aridity_index


def estimate_n_from_climate(
    P: float,
    PET: float
) -> float:
    """
    基于气候条件估算参数n的初始值
    
    经验关系（基于全球流域统计）:
    - 湿润区（PET/P < 1）: n ≈ 2-3（产流容易）
    - 半干旱区（PET/P 1-2）: n ≈ 1.5-2.5
    - 干旱区（PET/P > 2）: n ≈ 1-2（产流困难）
    
    参数:
        P: 降水量 (mm/year)
        PET: 潜在蒸散发 (mm/year)
    
    返回:
        n_initial: 估算的参数n初值
    
    注意:
        这仅是粗略估算，实际应通过calibrate_parameter_n精确求解
    
    示例:
        >>> n_init = estimate_n_from_climate(P=800, PET=1200)
        >>> print(f"估算n初值: {n_init:.2f}")
    """
    aridity = calculate_aridity_index(PET, P)
    
    if aridity < 1.0:
        # 湿润区：高n值，蒸散发能力强
        n_initial = 2.5
    elif aridity < 2.0:
        # 半干旱区：中等n值
        n_initial = 2.0
    else:
        # 干旱区：低n值，产流为主
        n_initial = 1.5
    
    return n_initial
