"""
parameter_calibration.py

批量站点参数校准和归因分析模块

本模块实现Budyko框架的批量站点处理工作流，包括：
- 多站点参数n的反演校准
- 时段划分和参数演变分析
- 气候变化和土地利用变化的归因分解
- 不确定性评估和敏感性分析
- 结果汇总和可视化

主要类：
    ParameterCalibrator: 参数校准和归因分析的主控制器
    CalibrationResult: 单站点校准结果的数据容器
    AttributionResult: 归因分析结果的数据容器

技术特点：
    - 支持并行处理（multiprocessing）
    - 严格的物理约束检查（Q_n < P）
    - Bootstrap重采样不确定性分析
    - 灵活的时段划分策略

作者: Research Software Engineer
日期: 2025-01-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 导入项目内部模块
from .core_equations import BudykoModel, validate_water_balance, calculate_aridity_index


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    单站点参数校准结果的数据容器
    
    Attributes:
        station_id: 站点标识符
        n: 校准得到的流域景观参数
        P: 多年平均降水量 (mm/year)
        PET: 多年平均潜在蒸散发 (mm/year)
        Q_n: 多年平均天然径流 (mm/year)
        E: 计算得到的实际蒸散发 (mm/year)
        aridity_index: 干旱指数 (PET/P)
        calibration_error: 校准误差（相对误差，%）
        water_balance_valid: 水量平衡是否有效
        period: 时间段标识（如'1960-1985', '1986-2016'）
        convergence: 优化算法是否收敛
    """
    station_id: str
    n: float
    P: float
    PET: float
    Q_n: float
    E: float
    aridity_index: float
    calibration_error: float
    water_balance_valid: bool
    period: str = ""
    convergence: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'station_id': self.station_id,
            'n': self.n,
            'P': self.P,
            'PET': self.PET,
            'Q_n': self.Q_n,
            'E': self.E,
            'aridity_index': self.aridity_index,
            'calibration_error': self.calibration_error,
            'water_balance_valid': self.water_balance_valid,
            'period': self.period,
            'convergence': self.convergence,
            **self.metadata
        }


@dataclass
class AttributionResult:
    """
    归因分析结果的数据容器
    
    Attributes:
        station_id: 站点标识符
        delta_Q_obs: 观测径流变化量 (mm/year)
        delta_Q_n: 天然径流变化量 (mm/year)
        delta_Q_CCV: 气候变化贡献 (mm/year)
        delta_Q_LUCC: 土地利用变化贡献 (mm/year)
        delta_Q_WADR: 人类取用水贡献 (mm/year)
        C_CCV: 气候变化贡献率 (%)
        C_LUCC: 土地利用变化贡献率 (%)
        C_WADR: 人类取用水贡献率 (%)
        period_base: 基准期
        period_impact: 影响期
        elasticity: 弹性系数字典 {epsilon_P, epsilon_PET, epsilon_n}
    """
    station_id: str
    delta_Q_obs: float
    delta_Q_n: float
    delta_Q_CCV: float
    delta_Q_LUCC: float
    delta_Q_WADR: float
    C_CCV: float
    C_LUCC: float
    C_WADR: float
    period_base: str
    period_impact: str
    elasticity: Dict[str, float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'station_id': self.station_id,
            'delta_Q_obs': self.delta_Q_obs,
            'delta_Q_n': self.delta_Q_n,
            'delta_Q_CCV': self.delta_Q_CCV,
            'delta_Q_LUCC': self.delta_Q_LUCC,
            'delta_Q_WADR': self.delta_Q_WADR,
            'C_CCV': self.C_CCV,
            'C_LUCC': self.C_LUCC,
            'C_WADR': self.C_WADR,
            'period_base': self.period_base,
            'period_impact': self.period_impact,
            **{f'epsilon_{k}': v for k, v in self.elasticity.items()},
            **{f'uncertainty_{k}': v for k, v in self.uncertainty.items()}
        }


class ParameterCalibrator:
    """
    参数校准和归因分析的主控制器
    
    本类集成了模块1-4的功能，实现完整的Budyko归因分析工作流：
    1. 数据加载和预处理
    2. 批量站点参数校准
    3. 时段对比分析
    4. 弹性系数计算
    5. 归因分解
    6. 不确定性评估
    
    Attributes:
        budyko_model: BudykoModel实例
        change_point: 突变点年份（默认1986）
        min_valid_years: 每个时段的最小有效年份数
        epsilon: 数值平滑参数
    """
    
    def __init__(
        self,
        change_point: int = 1986,
        min_valid_years: int = 10,
        epsilon: float = 1e-10
    ):
        """
        初始化参数校准器
        
        Args:
            change_point: 时间序列突变点（默认1986年）
            min_valid_years: 每个时段的最小有效年份数
            epsilon: 数值平滑参数，防止除零和对数错误
        """
        self.budyko_model = BudykoModel(epsilon=epsilon)
        self.change_point = change_point
        self.min_valid_years = min_valid_years
        self.epsilon = epsilon
        
        logger.info(
            f"ParameterCalibrator initialized: change_point={change_point}, "
            f"min_valid_years={min_valid_years}"
        )
    
    def calibrate_single_station(
        self,
        station_id: str,
        P: float,
        PET: float,
        Q_n: float,
        period: str = "",
        method: str = 'brentq'
    ) -> Optional[CalibrationResult]:
        """
        对单个站点进行参数校准
        
        Args:
            station_id: 站点标识符
            P: 多年平均降水量 (mm/year)
            PET: 多年平均潜在蒸散发 (mm/year)
            Q_n: 多年平均天然径流 (mm/year)
            period: 时间段标识
            method: 数值求解方法 ('brentq' 或 'newton')
        
        Returns:
            CalibrationResult对象，如果校准失败则返回None
        """
        # 物理一致性检查
        try:
            validate_water_balance(P, Q_n)
        except ValueError as e:
            logger.warning(f"Station {station_id} ({period}): {str(e)}")
            return None
        
        # 参数校准
        try:
            n = self.budyko_model.calibrate_parameter_n(P, PET, Q_n, method=method)
        except Exception as e:
            logger.error(
                f"Station {station_id} ({period}): Calibration failed - {str(e)}"
            )
            return None
        
        # 计算实际蒸散发和干旱指数
        E = self.budyko_model.calculate_actual_ET(P, PET, n)
        aridity_index = calculate_aridity_index(PET, P)
        
        # 验证校准精度
        Q_n_simulated = self.budyko_model.calculate_naturalized_runoff(P, PET, n)
        calibration_error = abs(Q_n_simulated - Q_n) / Q_n * 100  # 相对误差 (%)
        
        result = CalibrationResult(
            station_id=station_id,
            n=n,
            P=P,
            PET=PET,
            Q_n=Q_n,
            E=E,
            aridity_index=aridity_index,
            calibration_error=calibration_error,
            water_balance_valid=True,
            period=period,
            convergence=True
        )
        
        logger.debug(
            f"Station {station_id} ({period}): n={n:.4f}, "
            f"aridity={aridity_index:.3f}, error={calibration_error:.4f}%"
        )
        
        return result
    
    def batch_calibrate_stations(
        self,
        data: pd.DataFrame,
        station_id_col: str = 'station_id',
        P_col: str = 'P',
        PET_col: str = 'PET',
        Q_n_col: str = 'Q_n',
        parallel: bool = False,
        n_workers: int = 4
    ) -> List[CalibrationResult]:
        """
        批量校准多个站点
        
        Args:
            data: 包含站点数据的DataFrame
            station_id_col: 站点ID列名
            P_col: 降水列名
            PET_col: PET列名
            Q_n_col: 天然径流列名
            parallel: 是否使用并行处理
            n_workers: 并行工作进程数
        
        Returns:
            CalibrationResult对象列表
        """
        results = []
        
        if parallel:
            logger.info(f"Starting parallel calibration with {n_workers} workers...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for _, row in data.iterrows():
                    future = executor.submit(
                        self.calibrate_single_station,
                        row[station_id_col],
                        row[P_col],
                        row[PET_col],
                        row[Q_n_col]
                    )
                    futures[future] = row[station_id_col]
                
                for future in as_completed(futures):
                    station_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Station {station_id}: Exception - {str(e)}")
        else:
            logger.info(f"Starting sequential calibration for {len(data)} stations...")
            for _, row in data.iterrows():
                result = self.calibrate_single_station(
                    row[station_id_col],
                    row[P_col],
                    row[PET_col],
                    row[Q_n_col]
                )
                if result is not None:
                    results.append(result)
        
        logger.info(
            f"Calibration completed: {len(results)}/{len(data)} stations successful"
        )
        return results
    
    def analyze_parameter_evolution(
        self,
        station_id: str,
        time_series: pd.DataFrame,
        P_col: str = 'P',
        PET_col: str = 'PET',
        Q_n_col: str = 'Q_n',
        year_col: str = 'year'
    ) -> Dict[str, CalibrationResult]:
        """
        分析参数n在不同时段的演变
        
        根据突变点将时间序列划分为两个时段（如1960-1985和1986-2016），
        分别校准参数n，用于检测下垫面特征的变化（LUCC信号）。
        
        Args:
            station_id: 站点标识符
            time_series: 包含年度数据的DataFrame
            P_col: 降水列名
            PET_col: PET列名
            Q_n_col: 天然径流列名
            year_col: 年份列名
        
        Returns:
            字典 {'period_1': CalibrationResult, 'period_2': CalibrationResult, 'full': CalibrationResult}
        """
        results = {}
        
        # 全时段校准
        P_full = time_series[P_col].mean()
        PET_full = time_series[PET_col].mean()
        Q_n_full = time_series[Q_n_col].mean()
        
        result_full = self.calibrate_single_station(
            station_id, P_full, PET_full, Q_n_full, period='full'
        )
        if result_full is not None:
            results['full'] = result_full
        
        # 划分时段
        period_1 = time_series[time_series[year_col] < self.change_point]
        period_2 = time_series[time_series[year_col] >= self.change_point]
        
        # 检查时段长度
        if len(period_1) < self.min_valid_years:
            logger.warning(
                f"Station {station_id}: Period 1 has insufficient data "
                f"({len(period_1)} < {self.min_valid_years} years)"
            )
        else:
            P_1 = period_1[P_col].mean()
            PET_1 = period_1[PET_col].mean()
            Q_n_1 = period_1[Q_n_col].mean()
            period_1_label = f"{period_1[year_col].min()}-{period_1[year_col].max()}"
            
            result_1 = self.calibrate_single_station(
                station_id, P_1, PET_1, Q_n_1, period=period_1_label
            )
            if result_1 is not None:
                results['period_1'] = result_1
        
        if len(period_2) < self.min_valid_years:
            logger.warning(
                f"Station {station_id}: Period 2 has insufficient data "
                f"({len(period_2)} < {self.min_valid_years} years)"
            )
        else:
            P_2 = period_2[P_col].mean()
            PET_2 = period_2[PET_col].mean()
            Q_n_2 = period_2[Q_n_col].mean()
            period_2_label = f"{period_2[year_col].min()}-{period_2[year_col].max()}"
            
            result_2 = self.calibrate_single_station(
                station_id, P_2, PET_2, Q_n_2, period=period_2_label
            )
            if result_2 is not None:
                results['period_2'] = result_2
        
        return results
    
    def calculate_attribution(
        self,
        station_id: str,
        period_1_data: CalibrationResult,
        period_2_data: CalibrationResult,
        Q_obs_1: float,
        Q_obs_2: float
    ) -> Optional[AttributionResult]:
        """
        计算径流变化的归因分解
        
        基于main.tex文档的Steps 1-5，将径流变化分解为：
        - CCV (气候变化和变率): 由ΔP和ΔPET导致
        - LUCC (土地利用/覆盖变化): 由Δn导致
        - WADR (人类取用水和调蓄): 由Q_obs - Q_n的差异导致
        
        Args:
            station_id: 站点标识符
            period_1_data: 基准期校准结果
            period_2_data: 影响期校准结果
            Q_obs_1: 基准期观测径流 (mm/year)
            Q_obs_2: 影响期观测径流 (mm/year)
        
        Returns:
            AttributionResult对象，如果计算失败则返回None
        """
        # 计算变化量
        delta_P = period_2_data.P - period_1_data.P
        delta_PET = period_2_data.PET - period_1_data.PET
        delta_n = period_2_data.n - period_1_data.n
        delta_Q_obs = Q_obs_2 - Q_obs_1
        delta_Q_n = period_2_data.Q_n - period_1_data.Q_n
        
        # 检查观测径流变化量是否过小
        if abs(delta_Q_obs) < 1.0:  # 阈值：1 mm/year
            logger.warning(
                f"Station {station_id}: Delta Q_obs too small ({delta_Q_obs:.2f} mm), "
                "contribution rates may be unreliable"
            )
        
        # 使用全时段或period_1的平均值计算弹性系数
        P_mean = period_1_data.P
        PET_mean = period_1_data.PET
        n_mean = period_1_data.n
        Q_n_mean = period_1_data.Q_n
        
        try:
            elasticity = self.budyko_model.calculate_elasticities(
                P_mean, PET_mean, n_mean
            )
        except Exception as e:
            logger.error(
                f"Station {station_id}: Elasticity calculation failed - {str(e)}"
            )
            return None
        
        # 计算归因贡献（main.tex equation 7）
        epsilon_P = elasticity['epsilon_P']
        epsilon_PET = elasticity['epsilon_PET']
        epsilon_n = elasticity['epsilon_n']
        
        delta_Q_CCV = (
            epsilon_P * (Q_n_mean / P_mean) * delta_P +
            epsilon_PET * (Q_n_mean / PET_mean) * delta_PET
        )
        
        delta_Q_LUCC = epsilon_n * (Q_n_mean / n_mean) * delta_n
        
        delta_Q_WADR = delta_Q_obs - delta_Q_n
        
        # 计算贡献率（main.tex equations 8-10）
        if abs(delta_Q_obs) >= 1.0:
            C_CCV = (delta_Q_CCV / delta_Q_obs) * 100
            C_LUCC = (delta_Q_LUCC / delta_Q_obs) * 100
            C_WADR = (delta_Q_WADR / delta_Q_obs) * 100
        else:
            C_CCV = np.nan
            C_LUCC = np.nan
            C_WADR = np.nan
        
        result = AttributionResult(
            station_id=station_id,
            delta_Q_obs=delta_Q_obs,
            delta_Q_n=delta_Q_n,
            delta_Q_CCV=delta_Q_CCV,
            delta_Q_LUCC=delta_Q_LUCC,
            delta_Q_WADR=delta_Q_WADR,
            C_CCV=C_CCV,
            C_LUCC=C_LUCC,
            C_WADR=C_WADR,
            period_base=period_1_data.period,
            period_impact=period_2_data.period,
            elasticity=elasticity
        )
        
        logger.debug(
            f"Station {station_id}: ΔQ_obs={delta_Q_obs:.2f} mm, "
            f"C_CCV={C_CCV:.1f}%, C_LUCC={C_LUCC:.1f}%, C_WADR={C_WADR:.1f}%"
        )
        
        return result
    
    def bootstrap_uncertainty(
        self,
        time_series: pd.DataFrame,
        P_col: str = 'P',
        PET_col: str = 'PET',
        Q_n_col: str = 'Q_n',
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        使用Bootstrap重采样估计参数不确定性
        
        Args:
            time_series: 时间序列数据
            P_col, PET_col, Q_n_col: 列名
            n_bootstrap: Bootstrap重采样次数
            confidence_level: 置信水平
        
        Returns:
            字典，包含各参数的置信区间 {'n': (lower, upper), 'E': (lower, upper), ...}
        """
        n_samples = len(time_series)
        n_values = []
        E_values = []
        Q_n_values = []
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = time_series.sample(n=n_samples, replace=True)
            
            P_boot = bootstrap_sample[P_col].mean()
            PET_boot = bootstrap_sample[PET_col].mean()
            Q_n_boot = bootstrap_sample[Q_n_col].mean()
            
            # 尝试校准
            try:
                validate_water_balance(P_boot, Q_n_boot)
                n_boot = self.budyko_model.calibrate_parameter_n(
                    P_boot, PET_boot, Q_n_boot
                )
                E_boot = self.budyko_model.calculate_actual_ET(P_boot, PET_boot, n_boot)
                
                n_values.append(n_boot)
                E_values.append(E_boot)
                Q_n_values.append(Q_n_boot)
            except:
                continue
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        uncertainty = {}
        if len(n_values) > 0:
            uncertainty['n'] = (
                np.percentile(n_values, lower_percentile),
                np.percentile(n_values, upper_percentile)
            )
            uncertainty['E'] = (
                np.percentile(E_values, lower_percentile),
                np.percentile(E_values, upper_percentile)
            )
            uncertainty['Q_n'] = (
                np.percentile(Q_n_values, lower_percentile),
                np.percentile(Q_n_values, upper_percentile)
            )
        
        logger.info(
            f"Bootstrap uncertainty estimation: {len(n_values)}/{n_bootstrap} "
            "successful resamples"
        )
        
        return uncertainty
    
    def export_results(
        self,
        calibration_results: List[CalibrationResult],
        output_path: str
    ):
        """
        导出校准结果到CSV文件
        
        Args:
            calibration_results: CalibrationResult对象列表
            output_path: 输出文件路径
        """
        df = pd.DataFrame([r.to_dict() for r in calibration_results])
        df.to_csv(output_path, index=False)
        logger.info(f"Calibration results exported to {output_path}")
    
    def export_attribution_results(
        self,
        attribution_results: List[AttributionResult],
        output_path: str
    ):
        """
        导出归因结果到CSV文件
        
        Args:
            attribution_results: AttributionResult对象列表
            output_path: 输出文件路径
        """
        df = pd.DataFrame([r.to_dict() for r in attribution_results])
        df.to_csv(output_path, index=False)
        logger.info(f"Attribution results exported to {output_path}")


def validate_time_series_quality(
    time_series: pd.DataFrame,
    value_cols: List[str],
    max_missing_fraction: float = 0.15
) -> Tuple[bool, str]:
    """
    验证时间序列数据质量
    
    Args:
        time_series: 时间序列DataFrame
        value_cols: 需要检查的数值列名列表
        max_missing_fraction: 最大允许缺失比例
    
    Returns:
        (is_valid, message): 是否有效及诊断信息
    """
    for col in value_cols:
        if col not in time_series.columns:
            return False, f"Missing required column: {col}"
        
        missing_fraction = time_series[col].isna().sum() / len(time_series)
        if missing_fraction > max_missing_fraction:
            return False, (
                f"Column {col} has too many missing values "
                f"({missing_fraction*100:.1f}% > {max_missing_fraction*100:.1f}%)"
            )
        
        # 检查负值
        if (time_series[col].dropna() < 0).any():
            return False, f"Column {col} contains negative values"
    
    return True, "Data quality check passed"


def calculate_ensemble_attribution(
    attribution_results: List[AttributionResult],
    method: str = 'mean'
) -> Dict[str, float]:
    """
    计算多个归因结果的集合统计
    
    用于综合多个站点或多个模型的归因结果，提供区域尺度的平均归因。
    
    Args:
        attribution_results: AttributionResult对象列表
        method: 统计方法 ('mean', 'median', 'weighted_mean')
    
    Returns:
        集合归因统计字典
    """
    if not attribution_results:
        return {}
    
    # 提取各贡献量
    C_CCV_list = [r.C_CCV for r in attribution_results if not np.isnan(r.C_CCV)]
    C_LUCC_list = [r.C_LUCC for r in attribution_results if not np.isnan(r.C_LUCC)]
    C_WADR_list = [r.C_WADR for r in attribution_results if not np.isnan(r.C_WADR)]
    
    if method == 'mean':
        ensemble = {
            'C_CCV_mean': np.mean(C_CCV_list) if C_CCV_list else np.nan,
            'C_LUCC_mean': np.mean(C_LUCC_list) if C_LUCC_list else np.nan,
            'C_WADR_mean': np.mean(C_WADR_list) if C_WADR_list else np.nan,
            'C_CCV_std': np.std(C_CCV_list) if C_CCV_list else np.nan,
            'C_LUCC_std': np.std(C_LUCC_list) if C_LUCC_list else np.nan,
            'C_WADR_std': np.std(C_WADR_list) if C_WADR_list else np.nan,
            'n_stations': len(attribution_results)
        }
    elif method == 'median':
        ensemble = {
            'C_CCV_median': np.median(C_CCV_list) if C_CCV_list else np.nan,
            'C_LUCC_median': np.median(C_LUCC_list) if C_LUCC_list else np.nan,
            'C_WADR_median': np.median(C_WADR_list) if C_WADR_list else np.nan,
            'n_stations': len(attribution_results)
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble
