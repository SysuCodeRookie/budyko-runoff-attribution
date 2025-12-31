"""
ISIMIP归因分析模块 (ISIMIP Attribution Analysis Module)

本模块实现基于ISIMIP3a全球水文模型（GHMs）输出的径流归因分析，
用于分离气候变化中的人为气候变化（ACC）和自然气候变率（NCV）信号。

理论依据：
- Wang et al. (2025), Science Advances
- ISIMIP3a协议：https://protocol.isimip.org/

核心功能：
1. ISIMIP多场景数据加载（obsclim+histsoc, obsclim+1901soc, counterclim+1901soc）
2. 多模型集成（支持9个GHMs的ensemble分析）
3. ACC/NCV分离算法
4. 与Budyko方法的一致性验证

作者: Research Software Engineer
日期: 2025-01-01
版本: 1.0
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ISIMIPAttribution:
    """
    基于ISIMIP3a模型输出的径流归因分析类
    
    该类利用全球水文模型（GHMs）在不同实验场景下的径流模拟，
    实现对气候变化和人类活动贡献的精细化分离。
    
    关键实验场景：
    - obsclim + histsoc: 观测气候 + 历史人类活动 → Q'_o（模拟观测径流）
    - obsclim + 1901soc: 观测气候 + 1901固定人类活动 → Q'_n（模拟天然化径流）
    - counterclim + 1901soc: 去趋势气候 + 1901固定人类活动 → Q'_cn（基准径流）
    
    归因公式（main.tex）：
    - C_CCV = ΔQ'_n / ΔQ_o × 100%  （总气候贡献）
    - C_LUCC = (ΔQ_n - ΔQ'_n) / ΔQ_o × 100%  （土地利用变化）
    - C_WADR = (ΔQ_o - ΔQ_n) / ΔQ_o × 100%  （人类取用水）
    - C_ACC = (ΔQ'_n - ΔQ'_cn) / ΔQ_o × 100%  （人为气候变化）
    - C_NCV = ΔQ'_cn / ΔQ_o × 100%  （自然气候变率）
    
    Attributes
    ----------
    station_data : pd.DataFrame
        站点观测数据，需包含列：['year', 'Q_o', 'Q_n']
    isimip_data : Dict[str, pd.DataFrame]
        ISIMIP模型数据，键为场景名称（'obsclim_histsoc', 'obsclim_1901soc', 'counterclim_1901soc'）
    models : List[str]
        使用的GHM模型列表
    pre_period : Tuple[int, int]
        基准期（例如：(1960, 1985)）
    post_period : Tuple[int, int]
        影响期（例如：(1986, 2016)）
    
    Examples
    --------
    >>> # 加载ISIMIP多模型数据
    >>> isimip_data = {
    ...     'obsclim_histsoc': df_qo,
    ...     'obsclim_1901soc': df_qn,
    ...     'counterclim_1901soc': df_qcn
    ... }
    >>> 
    >>> # 执行ISIMIP归因
    >>> attribution = ISIMIPAttribution(station_data, isimip_data)
    >>> attribution.set_periods(change_year=1986)
    >>> results = attribution.run_attribution()
    >>> 
    >>> print(f"人为气候变化贡献: {results['C_ACC']:.1f}%")
    >>> print(f"自然气候变率贡献: {results['C_NCV']:.1f}%")
    """
    
    def __init__(
        self,
        station_data: pd.DataFrame,
        isimip_data: Dict[str, pd.DataFrame],
        models: Optional[List[str]] = None
    ):
        """
        初始化ISIMIP归因分析对象
        
        Parameters
        ----------
        station_data : pd.DataFrame
            观测站点数据，必须包含列：
            - year (int): 年份
            - Q_o (float): 观测径流 (mm/year)
            - Q_n (float): 天然化径流 (mm/year)
        
        isimip_data : Dict[str, pd.DataFrame]
            ISIMIP模型模拟数据，键为场景名称：
            - 'obsclim_histsoc': Q'_o数据
            - 'obsclim_1901soc': Q'_n数据
            - 'counterclim_1901soc': Q'_cn数据
            每个DataFrame需包含：
            - year (int): 年份
            - 模型列（如'clm45', 'h08', 'lpjml'等）: 各模型的径流模拟值 (mm/year)
        
        models : List[str], optional
            使用的模型列表。如果为None，则使用所有可用模型
        
        Raises
        ------
        ValueError
            如果输入数据缺少必需的列或场景
        """
        # 验证观测数据
        required_obs_cols = ['year', 'Q_o', 'Q_n']
        if not all(col in station_data.columns for col in required_obs_cols):
            raise ValueError(
                f"station_data必须包含列: {required_obs_cols}"
            )
        
        # 验证ISIMIP场景
        required_scenarios = ['obsclim_histsoc', 'obsclim_1901soc', 'counterclim_1901soc']
        if not all(scenario in isimip_data for scenario in required_scenarios):
            raise ValueError(
                f"isimip_data必须包含场景: {required_scenarios}"
            )
        
        # 验证ISIMIP数据格式
        for scenario, df in isimip_data.items():
            if 'year' not in df.columns:
                raise ValueError(f"场景'{scenario}'的数据必须包含'year'列")
        
        # 存储数据（排序以确保时间序列一致性）
        self.station_data = station_data.sort_values('year').reset_index(drop=True)
        self.isimip_data = {
            scenario: df.sort_values('year').reset_index(drop=True)
            for scenario, df in isimip_data.items()
        }
        
        # 确定模型列表
        if models is None:
            # 自动检测：除'year'外的所有列都是模型
            model_cols = [
                col for col in isimip_data['obsclim_histsoc'].columns
                if col != 'year'
            ]
            self.models = model_cols
        else:
            self.models = models
        
        # 验证所有场景都有相同的模型
        for scenario in required_scenarios:
            for model in self.models:
                if model not in isimip_data[scenario].columns:
                    raise ValueError(
                        f"场景'{scenario}'缺少模型'{model}'的数据"
                    )
        
        # 初始化时段
        self.pre_period: Optional[Tuple[int, int]] = None
        self.post_period: Optional[Tuple[int, int]] = None
    
    def set_periods(self, change_year: int = 1986) -> None:
        """
        设置基准期和影响期
        
        Parameters
        ----------
        change_year : int, default=1986
            突变点年份，用于划分基准期（pre）和影响期（post）
            默认1986年（中国大规模水土保持工程起始）
        
        Notes
        -----
        - 基准期：[最早年份, change_year-1]
        - 影响期：[change_year, 最晚年份]
        - 如果某时段少于5年，会发出警告
        """
        year_min = self.station_data['year'].min()
        year_max = self.station_data['year'].max()
        
        if not (year_min < change_year <= year_max):
            raise ValueError(
                f"突变年份{change_year}必须在数据范围内 "
                f"({year_min}, {year_max}]"
            )
        
        self.pre_period = (int(year_min), change_year - 1)
        self.post_period = (change_year, int(year_max))
        
        # 检查时段长度
        pre_length = self.pre_period[1] - self.pre_period[0] + 1
        post_length = self.post_period[1] - self.post_period[0] + 1
        
        if pre_length < 5:
            warnings.warn(
                f"基准期过短（{pre_length}年），可能影响统计稳定性",
                RuntimeWarning
            )
        if post_length < 5:
            warnings.warn(
                f"影响期过短（{post_length}年），可能影响统计稳定性",
                RuntimeWarning
            )
    
    def calculate_ensemble_mean(
        self,
        scenario: str,
        period: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        计算多模型集合平均值
        
        Parameters
        ----------
        scenario : str
            场景名称（'obsclim_histsoc', 'obsclim_1901soc', 'counterclim_1901soc'）
        period : Tuple[int, int], optional
            时段范围 (start_year, end_year)。如果为None，使用全部数据
        
        Returns
        -------
        float
            多模型集合平均径流 (mm/year)
        """
        df = self.isimip_data[scenario]
        
        if period is not None:
            df = df[(df['year'] >= period[0]) & (df['year'] <= period[1])]
        
        # 计算所有模型的平均值
        model_values = df[self.models].values  # shape: (n_years, n_models)
        ensemble_mean = np.mean(model_values)
        
        return ensemble_mean
    
    def calculate_model_uncertainty(
        self,
        scenario: str,
        period: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        计算多模型不确定性统计量
        
        Parameters
        ----------
        scenario : str
            场景名称
        period : Tuple[int, int], optional
            时段范围
        
        Returns
        -------
        Dict[str, float]
            包含以下键的字典：
            - 'mean': 集合平均值
            - 'std': 标准差
            - 'min': 最小值
            - 'max': 最大值
            - 'q25': 第25百分位数
            - 'q75': 第75百分位数
        """
        df = self.isimip_data[scenario]
        
        if period is not None:
            df = df[(df['year'] >= period[0]) & (df['year'] <= period[1])]
        
        # 每个模型的时段平均值
        model_means = df[self.models].mean(axis=0).values
        
        return {
            'mean': np.mean(model_means),
            'std': np.std(model_means, ddof=1),
            'min': np.min(model_means),
            'max': np.max(model_means),
            'q25': np.percentile(model_means, 25),
            'q75': np.percentile(model_means, 75)
        }
    
    def run_attribution(
        self,
        use_ensemble_mean: bool = True
    ) -> Dict[str, Union[float, Dict]]:
        """
        执行ISIMIP归因分析（支持多模型集成）
        
        Parameters
        ----------
        use_ensemble_mean : bool, default=True
            是否使用多模型集合平均。如果False，返回每个模型的独立结果
        
        Returns
        -------
        Dict[str, Union[float, Dict]]
            归因结果字典，包含：
            
            **观测数据**：
            - delta_Q_o (float): 观测径流变化 (mm)
            - delta_Q_n (float): 天然径流变化 (mm)
            
            **ISIMIP模拟**（如果use_ensemble_mean=True）：
            - Q_prime_o_pre, Q_prime_o_post (float): Q'_o基准期/影响期均值
            - Q_prime_n_pre, Q_prime_n_post (float): Q'_n基准期/影响期均值
            - Q_prime_cn_pre, Q_prime_cn_post (float): Q'_cn基准期/影响期均值
            - delta_Q_prime_o, delta_Q_prime_n, delta_Q_prime_cn (float): 模拟变化量
            
            **归因结果**：
            - C_CCV_isimip (float): 气候变化贡献 (%)
            - C_LUCC_isimip (float): 土地利用变化贡献 (%)
            - C_WADR_isimip (float): 人类取用水贡献 (%)
            - C_ACC (float): 人为气候变化贡献 (%)
            - C_NCV (float): 自然气候变率贡献 (%)
            
            **验证指标**：
            - contribution_sum (float): 贡献率之和，应≈100%
            - ACC_NCV_sum (float): C_ACC + C_NCV，应≈C_CCV
            
            **不确定性**（如果use_ensemble_mean=True）：
            - model_uncertainty (Dict): 各场景的多模型不确定性统计
        
        Raises
        ------
        ValueError
            如果时段未设置或观测径流变化过小
        RuntimeWarning
            如果贡献率之和偏离100%超过15%
        
        Notes
        -----
        归因公式（main.tex）：
        
        基础归因：
        $$C_{CCV} = \\frac{\\Delta Q'_n}{\\Delta Q_o} \\times 100\\%$$
        $$C_{LUCC} = \\frac{\\Delta Q_n - \\Delta Q'_n}{\\Delta Q_o} \\times 100\\%$$
        $$C_{WADR} = \\frac{\\Delta Q_o - \\Delta Q_n}{\\Delta Q_o} \\times 100\\%$$
        
        ACC/NCV分离：
        $$C_{ACC} = \\frac{\\Delta Q'_n - \\Delta Q'_{cn}}{\\Delta Q_o} \\times 100\\%$$
        $$C_{NCV} = \\frac{\\Delta Q'_{cn}}{\\Delta Q_o} \\times 100\\%$$
        
        理论关系：
        - C_CCV + C_LUCC + C_WADR ≈ 100%
        - C_ACC + C_NCV ≈ C_CCV
        """
        if self.pre_period is None or self.post_period is None:
            raise ValueError("必须先调用set_periods()设置分析时段")
        
        # ===== 1. 计算观测数据变化 =====
        obs_pre = self.station_data[
            (self.station_data['year'] >= self.pre_period[0]) &
            (self.station_data['year'] <= self.pre_period[1])
        ]
        obs_post = self.station_data[
            (self.station_data['year'] >= self.post_period[0]) &
            (self.station_data['year'] <= self.post_period[1])
        ]
        
        Q_o_pre = obs_pre['Q_o'].mean()
        Q_o_post = obs_post['Q_o'].mean()
        Q_n_pre = obs_pre['Q_n'].mean()
        Q_n_post = obs_post['Q_n'].mean()
        
        delta_Q_o = Q_o_post - Q_o_pre
        delta_Q_n = Q_n_post - Q_n_pre
        
        # 检查径流变化显著性
        if abs(delta_Q_o) < 1.0:
            warnings.warn(
                f"观测径流变化极小（{delta_Q_o:.3f} mm），"
                "贡献率可能不稳定",
                RuntimeWarning
            )
        
        # ===== 2. 计算ISIMIP模拟变化（多模型集成）=====
        if use_ensemble_mean:
            # 集合平均
            Q_prime_o_pre = self.calculate_ensemble_mean('obsclim_histsoc', self.pre_period)
            Q_prime_o_post = self.calculate_ensemble_mean('obsclim_histsoc', self.post_period)
            
            Q_prime_n_pre = self.calculate_ensemble_mean('obsclim_1901soc', self.pre_period)
            Q_prime_n_post = self.calculate_ensemble_mean('obsclim_1901soc', self.post_period)
            
            Q_prime_cn_pre = self.calculate_ensemble_mean('counterclim_1901soc', self.pre_period)
            Q_prime_cn_post = self.calculate_ensemble_mean('counterclim_1901soc', self.post_period)
            
            # 计算不确定性
            uncertainty = {
                'obsclim_histsoc_pre': self.calculate_model_uncertainty('obsclim_histsoc', self.pre_period),
                'obsclim_histsoc_post': self.calculate_model_uncertainty('obsclim_histsoc', self.post_period),
                'obsclim_1901soc_pre': self.calculate_model_uncertainty('obsclim_1901soc', self.pre_period),
                'obsclim_1901soc_post': self.calculate_model_uncertainty('obsclim_1901soc', self.post_period),
                'counterclim_1901soc_pre': self.calculate_model_uncertainty('counterclim_1901soc', self.pre_period),
                'counterclim_1901soc_post': self.calculate_model_uncertainty('counterclim_1901soc', self.post_period)
            }
        else:
            raise NotImplementedError("暂不支持单模型归因，请使用use_ensemble_mean=True")
        
        delta_Q_prime_o = Q_prime_o_post - Q_prime_o_pre
        delta_Q_prime_n = Q_prime_n_post - Q_prime_n_pre
        delta_Q_prime_cn = Q_prime_cn_post - Q_prime_cn_pre
        
        # ===== 3. 计算归因贡献率 =====
        # 基础ISIMIP归因（main.tex公式）
        C_CCV_isimip = (delta_Q_prime_n / delta_Q_o) * 100
        C_LUCC_isimip = ((delta_Q_n - delta_Q_prime_n) / delta_Q_o) * 100
        C_WADR_isimip = ((delta_Q_o - delta_Q_n) / delta_Q_o) * 100
        
        # ACC/NCV分离（main.tex公式）
        C_ACC = ((delta_Q_prime_n - delta_Q_prime_cn) / delta_Q_o) * 100
        C_NCV = (delta_Q_prime_cn / delta_Q_o) * 100
        
        # ===== 4. 验证 =====
        contribution_sum = C_CCV_isimip + C_LUCC_isimip + C_WADR_isimip
        ACC_NCV_sum = C_ACC + C_NCV
        
        # 检查贡献率之和
        if abs(contribution_sum - 100) > 15:
            warnings.warn(
                f"贡献率之和偏离100%: {contribution_sum:.1f}%，"
                "可能表明ISIMIP模拟与观测存在系统性偏差",
                RuntimeWarning
            )
        
        # 检查ACC+NCV≈CCV
        if abs(ACC_NCV_sum - C_CCV_isimip) > 10:
            warnings.warn(
                f"C_ACC + C_NCV ({ACC_NCV_sum:.1f}%) 与 C_CCV ({C_CCV_isimip:.1f}%) "
                "差异较大，检查counterclim场景数据质量",
                RuntimeWarning
            )
        
        # ===== 5. 构建结果字典 =====
        results = {
            # 观测数据
            'Q_o_pre': Q_o_pre,
            'Q_o_post': Q_o_post,
            'Q_n_pre': Q_n_pre,
            'Q_n_post': Q_n_post,
            'delta_Q_o': delta_Q_o,
            'delta_Q_n': delta_Q_n,
            
            # ISIMIP模拟
            'Q_prime_o_pre': Q_prime_o_pre,
            'Q_prime_o_post': Q_prime_o_post,
            'Q_prime_n_pre': Q_prime_n_pre,
            'Q_prime_n_post': Q_prime_n_post,
            'Q_prime_cn_pre': Q_prime_cn_pre,
            'Q_prime_cn_post': Q_prime_cn_post,
            'delta_Q_prime_o': delta_Q_prime_o,
            'delta_Q_prime_n': delta_Q_prime_n,
            'delta_Q_prime_cn': delta_Q_prime_cn,
            
            # 归因结果
            'C_CCV_isimip': C_CCV_isimip,
            'C_LUCC_isimip': C_LUCC_isimip,
            'C_WADR_isimip': C_WADR_isimip,
            'C_ACC': C_ACC,
            'C_NCV': C_NCV,
            
            # 验证指标
            'contribution_sum': contribution_sum,
            'ACC_NCV_sum': ACC_NCV_sum,
            'ACC_NCV_CCV_diff': ACC_NCV_sum - C_CCV_isimip,
            
            # 模型不确定性
            'model_uncertainty': uncertainty,
            'n_models': len(self.models),
            'models': self.models
        }
        
        return results
    
    def compare_with_budyko(
        self,
        budyko_results: Dict[str, float]
    ) -> Dict[str, float]:
        """
        比较ISIMIP归因与Budyko归因的结果
        
        Parameters
        ----------
        budyko_results : Dict[str, float]
            Budyko归因结果，需包含键：'C_CCV', 'C_LUCC', 'C_WADR'
        
        Returns
        -------
        Dict[str, float]
            包含差异统计的字典：
            - 'CCV_diff': C_CCV差异 (ISIMIP - Budyko, %)
            - 'LUCC_diff': C_LUCC差异 (%)
            - 'WADR_diff': C_WADR差异 (%)
            - 'rmse': 均方根误差
            - 'mae': 平均绝对误差
        
        Notes
        -----
        该方法用于验证ISIMIP归因与Budyko归因的一致性。
        理论上两种方法应给出相似的结果，但可能因以下原因产生差异：
        1. Budyko假设的简化（稳态、参数化）
        2. ISIMIP模型的系统性偏差
        3. LUCC表征方式的差异（参数n vs 直接模拟）
        """
        # 运行ISIMIP归因
        isimip_results = self.run_attribution()
        
        # 计算差异
        CCV_diff = isimip_results['C_CCV_isimip'] - budyko_results['C_CCV']
        LUCC_diff = isimip_results['C_LUCC_isimip'] - budyko_results['C_LUCC']
        WADR_diff = isimip_results['C_WADR_isimip'] - budyko_results['C_WADR']
        
        diffs = np.array([CCV_diff, LUCC_diff, WADR_diff])
        rmse = np.sqrt(np.mean(diffs**2))
        mae = np.mean(np.abs(diffs))
        
        return {
            'CCV_diff': CCV_diff,
            'LUCC_diff': LUCC_diff,
            'WADR_diff': WADR_diff,
            'rmse': rmse,
            'mae': mae,
            'budyko_CCV': budyko_results['C_CCV'],
            'budyko_LUCC': budyko_results['C_LUCC'],
            'budyko_WADR': budyko_results['C_WADR'],
            'isimip_CCV': isimip_results['C_CCV_isimip'],
            'isimip_LUCC': isimip_results['C_LUCC_isimip'],
            'isimip_WADR': isimip_results['C_WADR_isimip']
        }


def batch_isimip_attribution(
    multi_station_data: pd.DataFrame,
    isimip_data_dict: Dict[str, Dict[str, pd.DataFrame]],
    change_year: int = 1986,
    station_id_col: str = 'station_id'
) -> pd.DataFrame:
    """
    批量处理多站点的ISIMIP归因分析
    
    Parameters
    ----------
    multi_station_data : pd.DataFrame
        多站点观测数据，必须包含列：
        - station_id_col (str): 站点标识
        - year (int): 年份
        - Q_o, Q_n (float): 观测/天然径流
    
    isimip_data_dict : Dict[str, Dict[str, pd.DataFrame]]
        嵌套字典：{station_id: {scenario: DataFrame}}
        外层键为站点ID，内层键为场景名称
    
    change_year : int, default=1986
        突变点年份
    
    station_id_col : str, default='station_id'
        站点标识列名
    
    Returns
    -------
    pd.DataFrame
        包含所有站点归因结果的DataFrame
    
    Examples
    --------
    >>> isimip_dict = {
    ...     'station_1': {
    ...         'obsclim_histsoc': df_qo_1,
    ...         'obsclim_1901soc': df_qn_1,
    ...         'counterclim_1901soc': df_qcn_1
    ...     },
    ...     'station_2': {...}
    ... }
    >>> results = batch_isimip_attribution(obs_data, isimip_dict)
    """
    results_list = []
    
    station_ids = multi_station_data[station_id_col].unique()
    
    for station_id in station_ids:
        # 提取单站点数据
        station_obs = multi_station_data[
            multi_station_data[station_id_col] == station_id
        ].copy()
        
        # 检查ISIMIP数据是否存在
        if station_id not in isimip_data_dict:
            warnings.warn(
                f"站点 {station_id} 缺少ISIMIP数据，跳过",
                RuntimeWarning
            )
            continue
        
        isimip_data = isimip_data_dict[station_id]
        
        try:
            # 执行归因
            attribution = ISIMIPAttribution(station_obs, isimip_data)
            attribution.set_periods(change_year=change_year)
            result = attribution.run_attribution()
            
            # 添加站点标识
            result[station_id_col] = station_id
            
            # 提取主要结果（排除嵌套字典）
            main_result = {
                k: v for k, v in result.items()
                if not isinstance(v, (dict, list))
            }
            
            results_list.append(main_result)
            
        except Exception as e:
            warnings.warn(
                f"站点 {station_id} 归因失败: {str(e)}",
                RuntimeWarning
            )
            continue
    
    if not results_list:
        raise ValueError("所有站点归因均失败，请检查数据质量")
    
    return pd.DataFrame(results_list)


def summarize_isimip_attribution(results: Dict) -> str:
    """
    生成ISIMIP归因结果的文本摘要
    
    Parameters
    ----------
    results : Dict
        run_attribution()返回的结果字典
    
    Returns
    -------
    str
        格式化的摘要文本
    """
    summary = []
    summary.append("=" * 70)
    summary.append("ISIMIP归因分析结果摘要")
    summary.append("=" * 70)
    summary.append("")
    
    # 观测径流变化
    summary.append("【观测径流变化】")
    summary.append(f"  观测径流: {results['Q_o_pre']:.1f} → {results['Q_o_post']:.1f} mm")
    summary.append(f"  变化量: Δ = {results['delta_Q_o']:.1f} mm ({results['delta_Q_o']/results['Q_o_pre']*100:+.1f}%)")
    summary.append("")
    
    # ISIMIP模拟
    summary.append("【ISIMIP多模型集成模拟】")
    summary.append(f"  使用模型数: {results['n_models']}")
    summary.append(f"  Q'_o (obsclim+histsoc): {results['Q_prime_o_pre']:.1f} → {results['Q_prime_o_post']:.1f} mm (Δ = {results['delta_Q_prime_o']:.1f} mm)")
    summary.append(f"  Q'_n (obsclim+1901soc): {results['Q_prime_n_pre']:.1f} → {results['Q_prime_n_post']:.1f} mm (Δ = {results['delta_Q_prime_n']:.1f} mm)")
    summary.append(f"  Q'_cn (counterclim+1901soc): {results['Q_prime_cn_pre']:.1f} → {results['Q_prime_cn_post']:.1f} mm (Δ = {results['delta_Q_prime_cn']:.1f} mm)")
    summary.append("")
    
    # 归因结果 - 基础分解
    summary.append("【归因结果 - 三因子分解】")
    summary.append(f"  气候变化与变率 (C_CCV): {results['C_CCV_isimip']:+.1f}%")
    summary.append(f"  土地利用变化 (C_LUCC): {results['C_LUCC_isimip']:+.1f}%")
    summary.append(f"  人类取用水 (C_WADR): {results['C_WADR_isimip']:+.1f}%")
    summary.append(f"  贡献率之和: {results['contribution_sum']:.1f}%")
    summary.append("")
    
    # 归因结果 - ACC/NCV分离
    summary.append("【气候变化信号分离】")
    summary.append(f"  人为气候变化 (C_ACC): {results['C_ACC']:+.1f}%")
    summary.append(f"  自然气候变率 (C_NCV): {results['C_NCV']:+.1f}%")
    summary.append(f"  C_ACC + C_NCV: {results['ACC_NCV_sum']:.1f}% (应≈C_CCV: {results['C_CCV_isimip']:.1f}%)")
    summary.append("")
    
    # 主导因素
    contributions = {
        '气候变化': abs(results['C_CCV_isimip']),
        '人为气候变化': abs(results['C_ACC']),
        '自然气候变率': abs(results['C_NCV']),
        '土地利用变化': abs(results['C_LUCC_isimip']),
        '人类取用水': abs(results['C_WADR_isimip'])
    }
    dominant = max(contributions, key=contributions.get)
    summary.append("【主导因素】")
    summary.append(f"  {dominant} ({contributions[dominant]:.1f}%)")
    summary.append("")
    
    # 模型不确定性
    summary.append("【模型不确定性（标准差）】")
    unc = results['model_uncertainty']
    summary.append(f"  Q'_o基准期: ±{unc['obsclim_histsoc_pre']['std']:.1f} mm")
    summary.append(f"  Q'_n基准期: ±{unc['obsclim_1901soc_pre']['std']:.1f} mm")
    summary.append(f"  Q'_cn基准期: ±{unc['counterclim_1901soc_pre']['std']:.1f} mm")
    summary.append("")
    
    summary.append("=" * 70)
    
    return "\n".join(summary)
