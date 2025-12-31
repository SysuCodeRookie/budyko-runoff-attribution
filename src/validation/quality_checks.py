"""
物理一致性检验与质量控制模块

根据"代码撰写总体建议"第四章要求实现的质量检查功能。
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class QualityChecker:
    """
    数据质量和物理一致性检验类
    
    实现以下检查功能：
    1. 水量平衡检查
    2. 径流系数合理性检查
    3. Budyko参数范围检查
    4. 弹性系数符号合理性检查
    """
    
    @staticmethod
    def check_water_balance(P: float, Q: float, E: float, 
                          tolerance: float = 0.1) -> Tuple[bool, str]:
        """
        检查水量平衡: |P - Q - E| / P < tolerance
        
        在多年尺度上，假设地下水储量变化可忽略，则：
        P = Q + E (降水 = 径流 + 蒸散发)
        
        Parameters
        ----------
        P : float
            降水量 (mm/year)
        Q : float
            径流量 (mm/year)
        E : float
            蒸散发量 (mm/year)
        tolerance : float, optional
            相对误差容忍度，默认0.1 (10%)
            
        Returns
        -------
        is_valid : bool
            是否通过检验
        message : str
            检验信息
            
        Examples
        --------
        >>> QualityChecker.check_water_balance(800, 200, 600, tolerance=0.1)
        (True, "通过检验: 水量平衡误差 0.00%")
        
        >>> QualityChecker.check_water_balance(800, 300, 600, tolerance=0.1)
        (False, "警告: 水量平衡误差 12.50% 超过阈值 10.00%")
        """
        if P <= 0:
            return False, f"错误: 降水量 P={P:.2f} 必须为正值"
        
        residual = abs(P - Q - E)
        relative_error = residual / P
        
        if relative_error < tolerance:
            return True, f"通过检验: 水量平衡误差 {relative_error*100:.2f}%"
        else:
            return False, f"警告: 水量平衡误差 {relative_error*100:.2f}% 超过阈值 {tolerance*100:.2f}%"
    
    @staticmethod
    def check_runoff_ratio(Q: float, P: float, 
                          min_ratio: float = 0.05, 
                          max_ratio: float = 0.95) -> Tuple[bool, str]:
        """
        检查径流系数合理性: min_ratio < Q/P < max_ratio
        
        径流系数（Runoff Ratio）= Q/P，表示降水中转化为径流的比例。
        - 过低 (< 0.05): 可能存在数据错误或流域极端干旱
        - 过高 (> 0.95): 违反物理意义，蒸散发几乎为零
        
        Parameters
        ----------
        Q : float
            径流量 (mm/year)
        P : float
            降水量 (mm/year)
        min_ratio : float, optional
            最小允许径流系数，默认0.05
        max_ratio : float, optional
            最大允许径流系数，默认0.95
            
        Returns
        -------
        is_valid : bool
            是否通过检验
        message : str
            检验信息
            
        Notes
        -----
        该检查用于：
        - 剔除数据质量问题的站点
        - 识别Budyko方程不适用的极端情况
        - 预先筛选可进行参数反演的样本
        """
        if P <= 0:
            return False, f"错误: 降水量 P={P:.2f} 必须为正值"
        
        if Q < 0:
            return False, f"错误: 径流量 Q={Q:.2f} 不能为负值"
        
        ratio = Q / P
        
        if ratio > 1.0:
            return False, f"错误: 径流系数 {ratio:.3f} > 1.0，违反水量平衡"
        
        if ratio < min_ratio:
            return False, f"警告: 径流系数 {ratio:.3f} < {min_ratio:.3f}，可能数据异常或极端干旱"
        
        if ratio > max_ratio:
            return False, f"警告: 径流系数 {ratio:.3f} > {max_ratio:.3f}，疑似数据错误"
        
        return True, f"通过检验: 径流系数 {ratio:.3f} 在合理范围内"
    
    @staticmethod
    def check_parameter_range(n: float, 
                             min_n: float = 0.1, 
                             max_n: float = 10.0) -> Tuple[bool, str]:
        """
        检查Budyko模型参数n的合理性: min_n < n < max_n
        
        参数n代表流域景观特征，综合反映土壤、植被、地形对水分分配的影响。
        - 理论上 n > 0
        - 经验上 0.1 < n < 10（大多数流域在 1.0 ~ 5.0）
        - n较大：蒸散发能力强（森林、湿润区）
        - n较小：产流能力强（城市、干旱区）
        
        Parameters
        ----------
        n : float
            Budyko模型参数
        min_n : float, optional
            最小允许值，默认0.1
        max_n : float, optional
            最大允许值，默认10.0
            
        Returns
        -------
        is_valid : bool
            是否通过检验
        message : str
            检验信息
            
        References
        ----------
        - Yang et al. (2008): 全球流域n值主要在1.0-5.0
        - Xu et al. (2013): 超出范围通常意味着模型不适用或数据问题
        """
        if not np.isfinite(n):
            return False, f"错误: 参数 n={n} 非有限值"
        
        if n <= 0:
            return False, f"错误: 参数 n={n:.3f} 必须为正值"
        
        if n < min_n:
            return False, f"警告: 参数 n={n:.3f} < {min_n}，超出经验范围"
        
        if n > max_n:
            return False, f"警告: 参数 n={n:.3f} > {max_n}，超出经验范围"
        
        return True, f"通过检验: 参数 n={n:.3f} 在合理范围内"
    
    @staticmethod
    def check_elasticity_signs(eps_P: float, 
                               eps_PET: float, 
                               eps_n: float) -> Tuple[bool, str]:
        """
        检查弹性系数符号的合理性
        
        物理意义约束：
        - εP > 0:   降水增加 → 径流增加（正相关）
        - εPET < 0: 蒸散发需求增加 → 径流减少（负相关）
        - εn < 0:   下垫面截留能力增加 → 径流减少（负相关）
        
        Parameters
        ----------
        eps_P : float
            降水弹性系数
        eps_PET : float
            PET弹性系数
        eps_n : float
            流域参数弹性系数
            
        Returns
        -------
        is_valid : bool
            是否通过检验
        message : str
            检验信息
            
        Notes
        -----
        如果符号不符合物理意义，可能原因：
        1. 输入数据存在严重错误
        2. 模型假设在该流域不成立
        3. 数值计算出现异常
        """
        messages = []
        all_valid = True
        
        # 检查降水弹性系数
        if not np.isfinite(eps_P):
            all_valid = False
            messages.append(f"εP={eps_P} 非有限值")
        elif eps_P <= 0:
            all_valid = False
            messages.append(f"εP={eps_P:.3f} ≤ 0，违反物理意义（应为正值）")
        else:
            messages.append(f"εP={eps_P:.3f} ✓")
        
        # 检查PET弹性系数
        if not np.isfinite(eps_PET):
            all_valid = False
            messages.append(f"εPET={eps_PET} 非有限值")
        elif eps_PET >= 0:
            all_valid = False
            messages.append(f"εPET={eps_PET:.3f} ≥ 0，违反物理意义（应为负值）")
        else:
            messages.append(f"εPET={eps_PET:.3f} ✓")
        
        # 检查参数n弹性系数
        if not np.isfinite(eps_n):
            all_valid = False
            messages.append(f"εn={eps_n} 非有限值")
        elif eps_n >= 0:
            all_valid = False
            messages.append(f"εn={eps_n:.3f} ≥ 0，违反物理意义（应为负值）")
        else:
            messages.append(f"εn={eps_n:.3f} ✓")
        
        message = "弹性系数符号检查: " + "; ".join(messages)
        
        return all_valid, message
    
    @staticmethod
    def check_elasticity_sum(eps_P: float, 
                            eps_PET: float, 
                            min_sum: float = 0.5, 
                            max_sum: float = 1.5) -> Tuple[bool, str]:
        """
        检查弹性系数之和的合理性（软约束）
        
        对于某些Budyko公式形式，理论上有：εP + εPET = 1
        但Choudhury-Yang公式包含参数n，该关系可能略有偏差。
        此检查作为"软约束"，用于识别潜在的计算异常。
        
        Parameters
        ----------
        eps_P : float
            降水弹性系数
        eps_PET : float
            PET弹性系数
        min_sum : float, optional
            最小允许和，默认0.5
        max_sum : float, optional
            最大允许和，默认1.5
            
        Returns
        -------
        is_valid : bool
            是否通过检验
        message : str
            检验信息
            
        Notes
        -----
        该检查可选，不强制执行。主要用于质量控制和异常诊断。
        """
        elasticity_sum = eps_P + eps_PET
        
        if not np.isfinite(elasticity_sum):
            return False, f"错误: εP + εPET = {elasticity_sum} 非有限值"
        
        if elasticity_sum < min_sum or elasticity_sum > max_sum:
            return False, f"警告: εP + εPET = {elasticity_sum:.3f} 不在合理范围 [{min_sum}, {max_sum}]"
        
        return True, f"通过检验: εP + εPET = {elasticity_sum:.3f}"
    
    @staticmethod
    def comprehensive_check(P: float, Q: float, E: float, n: float,
                          eps_P: float, eps_PET: float, eps_n: float,
                          config: dict = None) -> Dict[str, Tuple[bool, str]]:
        """
        执行综合质量检查
        
        Parameters
        ----------
        P, Q, E, n : float
            水文变量和参数
        eps_P, eps_PET, eps_n : float
            弹性系数
        config : dict, optional
            配置参数字典，如果为None则使用默认值
            
        Returns
        -------
        results : dict
            各项检查的结果字典，格式为 {检查名称: (是否通过, 信息)}
            
        Examples
        --------
        >>> results = QualityChecker.comprehensive_check(
        ...     P=800, Q=200, E=600, n=2.5,
        ...     eps_P=2.0, eps_PET=-1.0, eps_n=-0.5
        ... )
        >>> all(r[0] for r in results.values())  # 检查是否全部通过
        True
        """
        if config is None:
            config = {
                'water_balance_tolerance': 0.1,
                'min_runoff_ratio': 0.05,
                'max_runoff_ratio': 0.95,
                'n_bounds': [0.1, 10.0],
                'elasticity_sum_range': [0.5, 1.5]
            }
        
        results = {}
        
        # 1. 水量平衡检查
        results['water_balance'] = QualityChecker.check_water_balance(
            P, Q, E, tolerance=config.get('water_balance_tolerance', 0.1)
        )
        
        # 2. 径流系数检查
        results['runoff_ratio'] = QualityChecker.check_runoff_ratio(
            Q, P,
            min_ratio=config.get('min_runoff_ratio', 0.05),
            max_ratio=config.get('max_runoff_ratio', 0.95)
        )
        
        # 3. 参数范围检查
        n_bounds = config.get('n_bounds', [0.1, 10.0])
        results['parameter_range'] = QualityChecker.check_parameter_range(
            n, min_n=n_bounds[0], max_n=n_bounds[1]
        )
        
        # 4. 弹性系数符号检查
        results['elasticity_signs'] = QualityChecker.check_elasticity_signs(
            eps_P, eps_PET, eps_n
        )
        
        # 5. 弹性系数和检查（可选）
        elasticity_range = config.get('elasticity_sum_range', [0.5, 1.5])
        results['elasticity_sum'] = QualityChecker.check_elasticity_sum(
            eps_P, eps_PET,
            min_sum=elasticity_range[0],
            max_sum=elasticity_range[1]
        )
        
        return results
    
    @staticmethod
    def log_check_results(results: Dict[str, Tuple[bool, str]], 
                         station_id: str = None) -> bool:
        """
        记录检查结果到日志
        
        Parameters
        ----------
        results : dict
            comprehensive_check返回的结果字典
        station_id : str, optional
            站点ID，用于日志标识
            
        Returns
        -------
        all_passed : bool
            是否所有检查都通过
        """
        prefix = f"站点 {station_id}: " if station_id else ""
        all_passed = True
        
        for check_name, (is_valid, message) in results.items():
            if is_valid:
                logger.info(f"{prefix}{check_name} - {message}")
            else:
                logger.warning(f"{prefix}{check_name} - {message}")
                all_passed = False
        
        return all_passed


# 数值稳定性工具函数（配合core_equations使用）

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
