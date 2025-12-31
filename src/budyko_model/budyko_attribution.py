"""
budyko_attribution.py

基于Budyko框架的径流归因分析模块

本模块实现完整的归因分析工作流，量化气候变化（CCV）、土地利用/覆盖变化（LUCC）
和人类取用水及水库调蓄（WADR）对径流变化的相对贡献。

核心功能：
1. 时段划分（基准期/影响期）
2. 参数n的率定（全时段、分时段）
3. 弹性系数计算
4. 径流变化归因分解
5. 贡献率计算与结果验证

理论依据：
- main.tex equations (1)-(10)：Budyko-Choudhury-Yang框架
- 6步归因流程（Step 1-6）

作者: Research Software Engineer
日期: 2025-01-01
版本: 1.0
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple, Union

from .core_equations import BudykoModel
from .elasticity_solver import (
    calculate_elasticity_P,
    calculate_elasticity_PET,
    calculate_elasticity_n,
    calculate_all_elasticities,
    validate_elasticity_signs
)


class BudykoAttribution:
    """
    Budyko归因分析主类
    
    实现基于Budyko假设的径流变化归因分析，遵循main.tex中描述的6步流程：
    
    Step 1: 全时段参数n率定（用于计算弹性系数）
    Step 2: 计算弹性系数（εP, εPET, εn）
    Step 3: 分时段参数n率定（量化LUCC影响）
    Step 4: 计算各驱动因子贡献（绝对量，mm）
    Step 5: 计算贡献率（相对量，%）
    Step 6: 模型验证与结果输出
    
    Parameters
    ----------
    station_data : pandas.DataFrame
        站点数据，必须包含以下列：
        - 'year': 年份（int）
        - 'P': 年均降水 (mm)
        - 'PET': 年均潜在蒸散发 (mm)
        - 'Qn': 年均天然径流 (mm)
        - 'Qo': 年均观测径流 (mm)
    
    Attributes
    ----------
    data : pandas.DataFrame
        原始输入数据
    pre_period : pandas.DataFrame
        基准期数据（默认：年份 < 1986）
    post_period : pandas.DataFrame
        影响期数据（默认：年份 >= 1986）
    change_year : int
        突变年份（默认：1986）
    
    Methods
    -------
    set_periods(change_year)
        设置基准期和影响期的分界点
    run_attribution()
        执行完整的6步归因分析
    validate_data_quality()
        检查输入数据的物理一致性
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'year': range(1960, 2017),
    ...     'P': np.random.uniform(500, 1000, 57),
    ...     'PET': np.random.uniform(800, 1200, 57),
    ...     'Qn': np.random.uniform(100, 300, 57),
    ...     'Qo': np.random.uniform(80, 280, 57)
    ... })
    >>> 
    >>> attribution = BudykoAttribution(data)
    >>> attribution.set_periods(change_year=1986)
    >>> results = attribution.run_attribution()
    >>> 
    >>> print(f"气候变化贡献: {results['C_CCV']:.1f}%")
    >>> print(f"土地利用变化贡献: {results['C_LUCC']:.1f}%")
    >>> print(f"人类取用水贡献: {results['C_WADR']:.1f}%")
    
    References
    ----------
    .. [1] Wang et al. (2025). China's nationwide streamflow decline driven 
           by landscape changes and human interventions. Science Advances.
    .. [2] Liu et al. (2019). Multimodel assessments of human and climate 
           impacts on mean annual streamflow in China. HESS.
    """
    
    def __init__(self, station_data: pd.DataFrame):
        """
        初始化归因分析对象
        
        Parameters
        ----------
        station_data : pandas.DataFrame
            必须包含 ['year', 'P', 'PET', 'Qn', 'Qo'] 列
        
        Raises
        ------
        ValueError
            如果必需列缺失
        """
        # 验证必需列
        required_cols = ['year', 'P', 'PET', 'Qn', 'Qo']
        missing_cols = [col for col in required_cols if col not in station_data.columns]
        
        if missing_cols:
            raise ValueError(
                f"输入数据缺少必需列: {missing_cols}。"
                f"必须包含: {required_cols}"
            )
        
        # 复制数据并排序
        self.data = station_data.copy().sort_values('year').reset_index(drop=True)
        
        # 初始化时段属性
        self.pre_period: Optional[pd.DataFrame] = None
        self.post_period: Optional[pd.DataFrame] = None
        self.change_year: Optional[int] = None
        
    def set_periods(self, change_year: int = 1986) -> None:
        """
        设置基准期（Pre-Change）和影响期（Post-Change）的分界点
        
        Parameters
        ----------
        change_year : int, optional
            突变年份，默认1986（中国大规模水土保持工程起始年）
        
        Examples
        --------
        >>> attribution.set_periods(change_year=1990)  # 自定义分界点
        """
        self.change_year = change_year
        self.pre_period = self.data[self.data['year'] < change_year].copy()
        self.post_period = self.data[self.data['year'] >= change_year].copy()
        
        # 验证时段长度
        if len(self.pre_period) < 5:
            warnings.warn(
                f"基准期数据点过少 (n={len(self.pre_period)})，"
                f"建议至少5年数据以确保统计稳定性",
                UserWarning
            )
        
        if len(self.post_period) < 5:
            warnings.warn(
                f"影响期数据点过少 (n={len(self.post_period)})，"
                f"建议至少5年数据以确保统计稳定性",
                UserWarning
            )
    
    def validate_data_quality(self) -> Tuple[bool, str]:
        """
        检查输入数据的物理一致性
        
        验证项：
        1. 无负值（P, PET, Qn, Qo > 0）
        2. 水量平衡合理性（Qn <= P）
        3. 人类影响合理性（Qo <= Qn，通常）
        
        Returns
        -------
        is_valid : bool
            数据是否通过验证
        message : str
            验证结果详细信息
        
        Examples
        --------
        >>> is_valid, msg = attribution.validate_data_quality()
        >>> if not is_valid:
        ...     print(f"数据质量警告: {msg}")
        """
        issues = []
        
        # 检查负值
        for col in ['P', 'PET', 'Qn', 'Qo']:
            if (self.data[col] < 0).any():
                n_negative = (self.data[col] < 0).sum()
                issues.append(f"{col}存在{n_negative}个负值")
        
        # 检查水量平衡（Qn不应超过P）
        invalid_balance = self.data['Qn'] > self.data['P']
        if invalid_balance.any():
            n_invalid = invalid_balance.sum()
            issues.append(
                f"{n_invalid}年的天然径流超过降水 (Qn > P)，违反水量平衡"
            )
        
        # 检查人类影响方向（通常Qo <= Qn，但可能有调水）
        Qo_exceed_Qn = self.data['Qo'] > self.data['Qn']
        if Qo_exceed_Qn.any():
            n_exceed = Qo_exceed_Qn.sum()
            warnings.warn(
                f"{n_exceed}年的观测径流超过天然径流 (Qo > Qn)，"
                f"可能存在跨流域调水或数据质量问题",
                UserWarning
            )
        
        if issues:
            return False, "; ".join(issues)
        else:
            return True, "数据质量检查通过"
    
    def run_attribution(
        self,
        epsilon: float = 1e-10,
        strict_validation: bool = False
    ) -> Dict[str, Union[float, Dict]]:
        """
        执行完整的6步Budyko归因分析
        
        遵循main.tex中描述的标准流程：
        
        Step 1: 全时段参数n率定
        -------
        使用1960-2016全时段数据反演参数n，作为流域固有特征的代表。
        
        Step 2: 计算弹性系数
        -------
        基于全时段均值计算εP, εPET, εn，用于后续归因计算。
        
        Step 3: 分时段参数n率定
        -------
        分别为基准期和影响期反演n值，其差值Δn反映土地利用变化。
        
        Step 4: 计算各驱动因子贡献（绝对量）
        -------
        - ΔQn,CCV = εP × (Qn/P) × ΔP + εPET × (Qn/PET) × ΔPET
        - ΔQn,LUCC = εn × (Qn/n) × Δn
        
        Step 5: 计算贡献率（相对量）
        -------
        - C_CCV = ΔQn,CCV / ΔQo × 100%
        - C_LUCC = ΔQn,LUCC / ΔQo × 100%
        - C_WADR = (ΔQo - ΔQn) / ΔQo × 100%
        
        Step 6: 模型验证
        -------
        验证Budyko模拟的径流变化与观测的一致性。
        
        Parameters
        ----------
        epsilon : float, optional
            数值稳定性参数，默认1e-10
        strict_validation : bool, optional
            是否进行严格的弹性系数验证，默认False
        
        Returns
        -------
        results : dict
            包含以下键值的归因结果：
            
            - 'n_full' (float): 全时段率定的参数n
            - 'n_pre' (float): 基准期参数n
            - 'n_post' (float): 影响期参数n
            - 'delta_n' (float): 参数n的变化量
            
            - 'P_full', 'PET_full', 'Qn_full' (float): 全时段均值
            - 'P_pre', 'PET_pre', 'Qn_pre' (float): 基准期均值
            - 'P_post', 'PET_post', 'Qn_post' (float): 影响期均值
            
            - 'delta_P', 'delta_PET' (float): 气候要素变化量
            - 'delta_Qo' (float): 观测径流变化量 (mm)
            - 'delta_Qn' (float): 天然径流变化量 (mm)
            
            - 'delta_Qn_CCV' (float): 气候变化贡献 (mm)
            - 'delta_Qn_LUCC' (float): 土地利用变化贡献 (mm)
            - 'delta_Qn_total' (float): Budyko模拟总变化 (mm)
            
            - 'C_CCV' (float): 气候变化贡献率 (%)
            - 'C_LUCC' (float): 土地利用变化贡献率 (%)
            - 'C_WADR' (float): 人类取用水贡献率 (%)
            
            - 'elasticity' (dict): 弹性系数
                - 'eps_P': 降水弹性系数
                - 'eps_PET': PET弹性系数
                - 'eps_n': 参数n弹性系数
            
            - 'validation' (dict): 验证指标
                - 'simulated_vs_observed': ΔQ̂n vs ΔQn的误差
                - 'contribution_sum': 贡献率之和（应接近100%）
        
        Raises
        ------
        ValueError
            如果时段未设置或参数率定失败
        RuntimeWarning
            如果径流变化过小，贡献率计算不稳定
        
        Examples
        --------
        >>> results = attribution.run_attribution()
        >>> 
        >>> # 查看贡献率
        >>> print(f"CCV: {results['C_CCV']:.1f}%")
        >>> print(f"LUCC: {results['C_LUCC']:.1f}%")
        >>> print(f"WADR: {results['C_WADR']:.1f}%")
        >>> 
        >>> # 查看弹性系数
        >>> print(results['elasticity'])
        >>> 
        >>> # 验证模型
        >>> print(f"贡献率之和: {results['validation']['contribution_sum']:.1f}%")
        """
        # 检查时段是否已设置
        if self.pre_period is None or self.post_period is None:
            raise ValueError(
                "必须先调用 set_periods() 设置基准期和影响期"
            )
        
        # ===================================================================
        # Step 1: 全时段参数n率定
        # ===================================================================
        P_full = self.data['P'].mean()
        PET_full = self.data['PET'].mean()
        Qn_full = self.data['Qn'].mean()
        
        # 使用BudykoModel进行参数率定
        budyko_model = BudykoModel(epsilon=epsilon)
        n_full = budyko_model.calibrate_parameter_n(
            P=P_full,
            PET=PET_full,
            Q_n=Qn_full
        )
        
        if n_full is None:
            raise ValueError(
                "全时段参数n率定失败。可能原因：\n"
                "1. 水量平衡违背（Qn > P）\n"
                "2. 输入数据异常\n"
                "请使用 validate_data_quality() 检查数据质量"
            )
        
        # ===================================================================
        # Step 2: 计算弹性系数（使用全时段均值）
        # ===================================================================
        elasticities = calculate_all_elasticities(
            P=P_full,
            PET=PET_full,
            n=n_full,
            epsilon=epsilon
        )
        
        eps_P = elasticities['epsilon_P']
        eps_PET = elasticities['epsilon_PET']
        eps_n = elasticities['epsilon_n']
        
        # 验证弹性系数符号
        is_valid_elasticity, elasticity_msg = validate_elasticity_signs(
            eps_P, eps_PET, eps_n, strict=strict_validation
        )
        
        if not is_valid_elasticity:
            warnings.warn(
                f"弹性系数验证失败: {elasticity_msg}。"
                f"εP={eps_P:.3f}, εPET={eps_PET:.3f}, εn={eps_n:.3f}",
                RuntimeWarning
            )
        
        # ===================================================================
        # Step 3: 分时段参数n率定
        # ===================================================================
        # 基准期
        P_pre = self.pre_period['P'].mean()
        PET_pre = self.pre_period['PET'].mean()
        Qn_pre = self.pre_period['Qn'].mean()
        Qo_pre = self.pre_period['Qo'].mean()
        
        n_pre = budyko_model.calibrate_parameter_n(
            P=P_pre,
            PET=PET_pre,
            Q_n=Qn_pre
        )
        
        if n_pre is None:
            raise ValueError("基准期参数n率定失败")
        
        # 影响期
        P_post = self.post_period['P'].mean()
        PET_post = self.post_period['PET'].mean()
        Qn_post = self.post_period['Qn'].mean()
        Qo_post = self.post_period['Qo'].mean()
        
        n_post = budyko_model.calibrate_parameter_n(
            P=P_post,
            PET=PET_post,
            Q_n=Qn_post
        )
        
        if n_post is None:
            raise ValueError("影响期参数n率定失败")
        
        # ===================================================================
        # Step 4: 计算各驱动因子的贡献（绝对量，mm）
        # ===================================================================
        # 气候要素变化量
        delta_P = P_post - P_pre
        delta_PET = PET_post - PET_pre
        delta_n = n_post - n_pre
        
        # 气候变化贡献（main.tex equation 7）
        delta_Qn_CCV = (
            eps_P * (Qn_full / P_full) * delta_P +
            eps_PET * (Qn_full / PET_full) * delta_PET
        )
        
        # 土地利用变化贡献（main.tex equation 7）
        delta_Qn_LUCC = eps_n * (Qn_full / n_full) * delta_n
        
        # Budyko模拟的总变化
        delta_Qn_total = delta_Qn_CCV + delta_Qn_LUCC
        
        # ===================================================================
        # Step 5: 计算贡献率（相对量，%）
        # ===================================================================
        delta_Qo = Qo_post - Qo_pre  # 观测径流变化
        delta_Qn = Qn_post - Qn_pre  # 天然径流变化
        
        # 检查变化量是否过小
        if abs(delta_Qo) < 1.0:  # 阈值：1mm
            warnings.warn(
                f"观测径流变化过小 (ΔQo = {delta_Qo:.2f} mm)，"
                f"贡献率计算可能不稳定。建议检查：\n"
                f"1. 是否选择了合适的突变年份？\n"
                f"2. 该站点是否确实发生了显著变化？",
                RuntimeWarning
            )
            
            # 仍然计算，但结果可能不可靠
            C_CCV = delta_Qn_CCV / delta_Qo * 100 if delta_Qo != 0 else np.nan
            C_LUCC = delta_Qn_LUCC / delta_Qo * 100 if delta_Qo != 0 else np.nan
            C_WADR = (delta_Qo - delta_Qn) / delta_Qo * 100 if delta_Qo != 0 else np.nan
        else:
            # 正常计算贡献率（main.tex equations 8-10）
            C_CCV = delta_Qn_CCV / delta_Qo * 100
            C_LUCC = delta_Qn_LUCC / delta_Qo * 100
            C_WADR = (delta_Qo - delta_Qn) / delta_Qo * 100
        
        # ===================================================================
        # Step 6: 模型验证与结果输出
        # ===================================================================
        # 验证Budyko模拟与观测的一致性
        simulated_vs_observed_error = delta_Qn_total - delta_Qn
        
        # 贡献率之和（应接近100%）
        contribution_sum = C_CCV + C_LUCC + C_WADR
        
        # 组装结果字典
        results = {
            # 参数n
            'n_full': n_full,
            'n_pre': n_pre,
            'n_post': n_post,
            'delta_n': delta_n,
            
            # 全时段均值
            'P_full': P_full,
            'PET_full': PET_full,
            'Qn_full': Qn_full,
            
            # 基准期均值
            'P_pre': P_pre,
            'PET_pre': PET_pre,
            'Qn_pre': Qn_pre,
            'Qo_pre': Qo_pre,
            
            # 影响期均值
            'P_post': P_post,
            'PET_post': PET_post,
            'Qn_post': Qn_post,
            'Qo_post': Qo_post,
            
            # 变化量
            'delta_P': delta_P,
            'delta_PET': delta_PET,
            'delta_Qo': delta_Qo,
            'delta_Qn': delta_Qn,
            
            # 归因结果（绝对量）
            'delta_Qn_CCV': delta_Qn_CCV,
            'delta_Qn_LUCC': delta_Qn_LUCC,
            'delta_Qn_total': delta_Qn_total,
            
            # 归因结果（相对量）
            'C_CCV': C_CCV,
            'C_LUCC': C_LUCC,
            'C_WADR': C_WADR,
            
            # 弹性系数
            'elasticity': {
                'eps_P': eps_P,
                'eps_PET': eps_PET,
                'eps_n': eps_n,
                'sum_P_PET': elasticities['sum_P_PET']
            },
            
            # 验证指标
            'validation': {
                'simulated_vs_observed': simulated_vs_observed_error,
                'contribution_sum': contribution_sum,
                'elasticity_valid': is_valid_elasticity,
                'elasticity_message': elasticity_msg
            }
        }
        
        return results


def batch_attribution(
    stations_data: pd.DataFrame,
    change_year: int = 1986,
    station_id_col: str = 'station_id',
    epsilon: float = 1e-10,
    strict_validation: bool = False
) -> pd.DataFrame:
    """
    批量处理多个站点的归因分析
    
    对包含多个站点数据的DataFrame进行批量归因计算，自动识别每个站点
    并分别执行完整的6步流程。
    
    Parameters
    ----------
    stations_data : pandas.DataFrame
        多站点数据，必须包含列：
        - station_id_col: 站点标识符（默认'station_id'）
        - 'year': 年份
        - 'P', 'PET', 'Qn', 'Qo': 水文气象变量
    change_year : int, optional
        突变年份，默认1986
    station_id_col : str, optional
        站点ID列名，默认'station_id'
    epsilon : float, optional
        数值稳定性参数，默认1e-10
    strict_validation : bool, optional
        是否严格验证，默认False
    
    Returns
    -------
    results_df : pandas.DataFrame
        归因结果汇总表，每行对应一个站点，包含所有归因指标
    
    Examples
    --------
    >>> # 假设有100个站点的数据
    >>> multi_station_data = pd.read_csv('all_stations.csv')
    >>> 
    >>> # 批量归因
    >>> results = batch_attribution(multi_station_data, change_year=1986)
    >>> 
    >>> # 查看结果统计
    >>> print(results[['station_id', 'C_CCV', 'C_LUCC', 'C_WADR']].describe())
    >>> 
    >>> # 筛选LUCC主导的站点
    >>> lucc_dominated = results[abs(results['C_LUCC']) > abs(results['C_CCV'])]
    >>> print(f"土地利用变化主导的站点: {len(lucc_dominated)}")
    """
    # 获取唯一站点列表
    station_ids = stations_data[station_id_col].unique()
    
    results_list = []
    
    for station_id in station_ids:
        # 提取单站点数据
        station_data = stations_data[
            stations_data[station_id_col] == station_id
        ].copy()
        
        try:
            # 创建归因对象
            attribution = BudykoAttribution(station_data)
            attribution.set_periods(change_year=change_year)
            
            # 执行归因
            result = attribution.run_attribution(
                epsilon=epsilon,
                strict_validation=strict_validation
            )
            
            # 添加站点ID
            result[station_id_col] = station_id
            results_list.append(result)
            
        except Exception as e:
            warnings.warn(
                f"站点 {station_id} 归因失败: {str(e)}",
                RuntimeWarning
            )
            continue
    
    # 转换为DataFrame
    if not results_list:
        raise ValueError("所有站点归因均失败，请检查数据质量")
    
    # 展开嵌套字典
    results_df = pd.DataFrame(results_list)
    
    # 展开elasticity子字典
    if 'elasticity' in results_df.columns:
        elasticity_df = pd.json_normalize(results_df['elasticity'])
        elasticity_df.columns = ['elasticity_' + col for col in elasticity_df.columns]
        results_df = pd.concat([results_df.drop('elasticity', axis=1), elasticity_df], axis=1)
    
    # 展开validation子字典
    if 'validation' in results_df.columns:
        validation_df = pd.json_normalize(results_df['validation'])
        validation_df.columns = ['validation_' + col for col in validation_df.columns]
        results_df = pd.concat([results_df.drop('validation', axis=1), validation_df], axis=1)
    
    return results_df


# ========================================================================
# 辅助函数
# ========================================================================

def summarize_attribution(results: Dict) -> str:
    """
    生成归因结果的文本摘要
    
    Parameters
    ----------
    results : dict
        run_attribution() 返回的结果字典
    
    Returns
    -------
    summary : str
        格式化的文本摘要
    
    Examples
    --------
    >>> results = attribution.run_attribution()
    >>> print(summarize_attribution(results))
    """
    summary_lines = [
        "=" * 70,
        "Budyko归因分析结果摘要",
        "=" * 70,
        "",
        "【参数n变化】",
        f"  全时段: n = {results['n_full']:.3f}",
        f"  基准期: n = {results['n_pre']:.3f}",
        f"  影响期: n = {results['n_post']:.3f}",
        f"  变化量: Δn = {results['delta_n']:.3f}",
        "",
        "【气候要素变化】",
        f"  降水: {results['P_pre']:.1f} → {results['P_post']:.1f} mm "
        f"(Δ = {results['delta_P']:+.1f} mm, {results['delta_P']/results['P_pre']*100:+.1f}%)",
        f"  PET: {results['PET_pre']:.1f} → {results['PET_post']:.1f} mm "
        f"(Δ = {results['delta_PET']:+.1f} mm, {results['delta_PET']/results['PET_pre']*100:+.1f}%)",
        "",
        "【径流变化】",
        f"  观测径流: {results['Qo_pre']:.1f} → {results['Qo_post']:.1f} mm "
        f"(Δ = {results['delta_Qo']:+.1f} mm, {results['delta_Qo']/results['Qo_pre']*100:+.1f}%)",
        f"  天然径流: {results['Qn_pre']:.1f} → {results['Qn_post']:.1f} mm "
        f"(Δ = {results['delta_Qn']:+.1f} mm, {results['delta_Qn']/results['Qn_pre']*100:+.1f}%)",
        "",
        "【弹性系数】",
        f"  εP (降水): {results['elasticity']['eps_P']:.3f}",
        f"  εPET (PET): {results['elasticity']['eps_PET']:.3f}",
        f"  εn (参数): {results['elasticity']['eps_n']:.3f}",
        "",
        "【归因结果 - 绝对贡献】",
        f"  气候变化 (CCV): {results['delta_Qn_CCV']:+.2f} mm",
        f"  土地利用 (LUCC): {results['delta_Qn_LUCC']:+.2f} mm",
        f"  Budyko模拟总变化: {results['delta_Qn_total']:+.2f} mm",
        "",
        "【归因结果 - 相对贡献率】",
        f"  气候变化 (C_CCV): {results['C_CCV']:.1f}%",
        f"  土地利用 (C_LUCC): {results['C_LUCC']:.1f}%",
        f"  人类取用水 (C_WADR): {results['C_WADR']:.1f}%",
        f"  贡献率之和: {results['validation']['contribution_sum']:.1f}%",
        "",
        "【主导因素】",
    ]
    
    # 判断主导因素
    abs_contributions = {
        '气候变化': abs(results['C_CCV']),
        '土地利用变化': abs(results['C_LUCC']),
        '人类取用水': abs(results['C_WADR'])
    }
    
    dominant_factor = max(abs_contributions, key=abs_contributions.get)
    dominant_percentage = abs_contributions[dominant_factor]
    
    summary_lines.append(f"  {dominant_factor}是径流变化的主导因素 ({dominant_percentage:.1f}%)")
    summary_lines.append("")
    summary_lines.append("=" * 70)
    
    return "\n".join(summary_lines)
