"""
PET计算器模块
===========

基于pyet库实现FAO-56 Penman-Monteith潜在蒸散发计算。

主要功能:
- FAO-56 Penman-Monteith方法计算参考作物蒸散发(ET0)
- 支持单点和时间序列计算
- 自动单位转换和数据验证
- 与ISIMIP气候数据无缝集成

作者: GitHub Copilot
日期: 2025-12-31
"""

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import warnings

# pyet库用于PET计算
try:
    import pyet
except ImportError:
    raise ImportError(
        "pyet库未安装。请运行: pip install pyet\n"
        "pyet是基于FAO-56标准的蒸散发计算库。"
    )


class PETCalculator:
    """
    潜在蒸散发(PET)计算器
    
    使用FAO-56 Penman-Monteith方法计算参考作物蒸散发(ET0)。
    该方法是国际公认的PET计算标准，广泛应用于水文学研究。
    
    参数
    ----
    latitude : float
        站点纬度 (度), 范围: -90 到 90
    elevation : float, optional
        站点海拔高度 (m), 用于大气压力计算, 默认0
    method : str, optional
        PET计算方法, 默认'pm' (Penman-Monteith)
        可选: 'pm', 'hargreaves', 'priestley_taylor'
    
    属性
    ----
    latitude : float
        站点纬度
    elevation : float
        海拔高度
    method : str
        计算方法
    metadata : dict
        计算器元数据
    
    示例
    ----
    >>> calculator = PETCalculator(latitude=30.5, elevation=50)
    >>> pet = calculator.calculate_fao56(
    ...     tmean=25.0,
    ...     tmax=30.0,
    ...     tmin=20.0,
    ...     rs=20.0,
    ...     rh=60.0,
    ...     uz=2.0
    ... )
    >>> print(f"PET: {pet:.2f} mm/day")
    """
    
    def __init__(
        self,
        latitude: float,
        elevation: float = 0.0,
        method: str = 'pm'
    ):
        """初始化PET计算器"""
        # 验证输入
        if not -90 <= latitude <= 90:
            raise ValueError(f"纬度必须在-90到90之间，当前值: {latitude}")
        
        if elevation < -500 or elevation > 9000:
            warnings.warn(
                f"海拔高度异常: {elevation} m。通常范围: -500至9000 m",
                UserWarning
            )
        
        valid_methods = ['pm', 'hargreaves', 'priestley_taylor']
        if method not in valid_methods:
            raise ValueError(
                f"不支持的方法: {method}。可选: {valid_methods}"
            )
        
        self.latitude = latitude
        self.elevation = elevation
        self.method = method
        
        self.metadata = {
            'latitude': latitude,
            'elevation': elevation,
            'method': method,
            'reference_crop': 'grass',  # FAO-56标准参考作物
            'created_at': pd.Timestamp.now().isoformat()
        }
    
    def calculate_fao56(
        self,
        tmean: Union[float, pd.Series, np.ndarray],
        tmax: Union[float, pd.Series, np.ndarray],
        tmin: Union[float, pd.Series, np.ndarray],
        rs: Union[float, pd.Series, np.ndarray],
        rh: Optional[Union[float, pd.Series, np.ndarray]] = None,
        uz: Optional[Union[float, pd.Series, np.ndarray]] = None,
        pressure: Optional[Union[float, pd.Series, np.ndarray]] = None
    ) -> Union[float, pd.Series]:
        """
        使用FAO-56 Penman-Monteith方法计算PET
        
        这是联合国粮农组织(FAO)推荐的标准方法，基于能量平衡和空气动力学
        原理，计算假设参考作物(短草)的蒸散发量。
        
        参数
        ----
        tmean : float or Series or ndarray
            日平均气温 (°C)
        tmax : float or Series or ndarray
            日最高气温 (°C)
        tmin : float or Series or ndarray
            日最低气温 (°C)
        rs : float or Series or ndarray
            太阳辐射 (MJ/m²/day)
        rh : float or Series or ndarray, optional
            相对湿度 (%), 如果未提供则使用默认值70%
        uz : float or Series or ndarray, optional
            2米高度风速 (m/s), 如果未提供则使用默认值2.0 m/s
        pressure : float or Series or ndarray, optional
            大气压力 (kPa), 如果未提供则根据海拔高度计算
        
        返回
        ----
        pet : float or Series
            潜在蒸散发 (mm/day)
        
        注意
        ----
        - 所有气温必须为摄氏度(°C)
        - 太阳辐射必须为MJ/m²/day
        - 返回的PET单位为mm/day，年总量需乘以365.25
        
        参考文献
        --------
        Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998.
        Crop evapotranspiration - Guidelines for computing crop water requirements.
        FAO Irrigation and drainage paper 56. FAO, Rome.
        """
        # 数据验证
        self._validate_inputs(tmean, tmax, tmin, rs, rh, uz)
        
        # 转换为pandas Series以统一处理
        is_scalar = np.isscalar(tmean)
        
        if is_scalar:
            # 标量输入，创建单日Series
            index = pd.date_range('2000-01-01', periods=1, freq='D')
            tmean = pd.Series([tmean], index=index)
            tmax = pd.Series([tmax], index=index)
            tmin = pd.Series([tmin], index=index)
            rs = pd.Series([rs], index=index)
            
            if rh is not None:
                rh = pd.Series([rh], index=index)
            if uz is not None:
                uz = pd.Series([uz], index=index)
        else:
            # 确保有时间索引
            if not isinstance(tmean, pd.Series):
                tmean = pd.Series(tmean)
                tmax = pd.Series(tmax)
                tmin = pd.Series(tmin)
                rs = pd.Series(rs)
                
                if rh is not None:
                    rh = pd.Series(rh)
                if uz is not None:
                    uz = pd.Series(uz)
        
        # 设置默认值
        if rh is None:
            rh = pd.Series(70.0, index=tmean.index)  # 默认70%
        if uz is None:
            uz = pd.Series(2.0, index=tmean.index)   # 默认2 m/s
        
        # 计算大气压力（如果未提供）
        if pressure is None:
            # FAO-56公式：P = 101.3 * [(293 - 0.0065*z) / 293]^5.26
            pressure = 101.3 * ((293 - 0.0065 * self.elevation) / 293) ** 5.26
        
        try:
            # 调用pyet的FAO-56 Penman-Monteith方法
            pet = pyet.pm_fao56(
                tmean=tmean,
                tmax=tmax,
                tmin=tmin,
                wind=uz,
                rs=rs,
                rh=rh,
                pressure=pressure,
                elevation=self.elevation,
                lat=self.latitude
            )
            
            # 如果是标量输入，返回标量
            if is_scalar:
                return float(pet.iloc[0])
            
            return pet
            
        except Exception as e:
            raise RuntimeError(
                f"PET计算失败: {str(e)}\n"
                f"请检查输入数据的单位和范围是否正确。"
            ) from e
    
    def calculate_from_climate_data(
        self,
        climate_data: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        从climate_processor处理后的数据计算PET
        
        这是一个便捷方法，直接接受ClimateDataProcessor的输出字典。
        
        参数
        ----
        climate_data : dict
            包含气候变量的字典，键应包含:
            - 'tas' 或 'tmean': 平均气温 (°C)
            - 'tasmax' 或 'tmax': 最高气温 (°C)
            - 'tasmin' 或 'tmin': 最低气温 (°C)
            - 'rsds' 或 'rs': 太阳辐射 (MJ/m²/day)
            - 'hurs' 或 'rh': 相对湿度 (%), 可选
            - 'sfcWind' 或 'uz': 风速 (m/s), 可选
        
        返回
        ----
        pet : Series
            日尺度PET时间序列 (mm/day)
        
        示例
        ----
        >>> from src.data_preprocessing.climate_processor import ClimateDataProcessor
        >>> climate_data = ClimateDataProcessor.batch_process_variables(...)
        >>> calculator = PETCalculator(latitude=30.5)
        >>> pet = calculator.calculate_from_climate_data(climate_data)
        """
        # 提取变量（支持不同的命名）
        tmean = climate_data.get('tas') if 'tas' in climate_data else climate_data.get('tmean')
        tmax = climate_data.get('tasmax') if 'tasmax' in climate_data else climate_data.get('tmax')
        tmin = climate_data.get('tasmin') if 'tasmin' in climate_data else climate_data.get('tmin')
        rs = climate_data.get('rsds') if 'rsds' in climate_data else climate_data.get('rs')
        rh = climate_data.get('hurs') if 'hurs' in climate_data else climate_data.get('rh')
        uz = climate_data.get('sfcWind') if 'sfcWind' in climate_data else climate_data.get('uz')
        
        # 验证必需变量
        missing = []
        if tmean is None:
            missing.append('tas/tmean')
        if tmax is None:
            missing.append('tasmax/tmax')
        if tmin is None:
            missing.append('tasmin/tmin')
        if rs is None:
            missing.append('rsds/rs')
        
        if missing:
            raise ValueError(
                f"缺少必需的气候变量: {', '.join(missing)}\n"
                f"可用变量: {list(climate_data.keys())}"
            )
        
        return self.calculate_fao56(
            tmean=tmean,
            tmax=tmax,
            tmin=tmin,
            rs=rs,
            rh=rh,
            uz=uz
        )
    
    def calculate_hargreaves(
        self,
        tmean: Union[float, pd.Series],
        tmax: Union[float, pd.Series],
        tmin: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """
        使用Hargreaves方法计算PET（简化方法）
        
        当缺少辐射、湿度、风速数据时，可使用此方法。
        仅需气温数据，但精度低于FAO-56方法。
        
        参数
        ----
        tmean, tmax, tmin : float or Series
            平均、最高、最低气温 (°C)
        
        返回
        ----
        pet : float or Series
            PET (mm/day)
        """
        is_scalar = np.isscalar(tmean)
        
        if is_scalar:
            index = pd.date_range('2000-01-01', periods=1, freq='D')
            tmean = pd.Series([tmean], index=index)
            tmax = pd.Series([tmax], index=index)
            tmin = pd.Series([tmin], index=index)
        
        pet = pyet.hargreaves(
            tmean=tmean,
            tmax=tmax,
            tmin=tmin,
            lat=self.latitude
        )
        
        if is_scalar:
            return float(pet.iloc[0])
        
        return pet
    
    def aggregate_to_annual(
        self,
        pet_daily: pd.Series,
        method: str = 'sum'
    ) -> pd.Series:
        """
        将日尺度PET聚合为年尺度
        
        参数
        ----
        pet_daily : Series
            日尺度PET (mm/day)
        method : str
            聚合方法，'sum' (默认) 或 'mean'
        
        返回
        ----
        pet_annual : Series
            年尺度PET (mm/year)
        """
        if not isinstance(pet_daily.index, pd.DatetimeIndex):
            raise ValueError("输入必须有DatetimeIndex")
        
        if method == 'sum':
            pet_annual = pet_daily.resample('YE').sum()
        elif method == 'mean':
            pet_annual = pet_daily.resample('YE').mean() * 365.25
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        # 简化索引为年份
        pet_annual.index = pet_annual.index.year
        pet_annual.name = 'PET'
        
        return pet_annual
    
    def _validate_inputs(
        self,
        tmean: Union[float, pd.Series, np.ndarray],
        tmax: Union[float, pd.Series, np.ndarray],
        tmin: Union[float, pd.Series, np.ndarray],
        rs: Union[float, pd.Series, np.ndarray],
        rh: Optional[Union[float, pd.Series, np.ndarray]],
        uz: Optional[Union[float, pd.Series, np.ndarray]]
    ):
        """验证输入数据的合理性"""
        # 转换为数组以便检查
        def to_array(x):
            if x is None:
                return None
            if np.isscalar(x):
                return np.array([x])
            if isinstance(x, pd.Series):
                return x.values
            return np.asarray(x)
        
        tmean_arr = to_array(tmean)
        tmax_arr = to_array(tmax)
        tmin_arr = to_array(tmin)
        rs_arr = to_array(rs)
        rh_arr = to_array(rh)
        uz_arr = to_array(uz)
        
        # 气温范围检查
        if np.any(tmean_arr < -50) or np.any(tmean_arr > 60):
            warnings.warn(
                "平均气温超出合理范围(-50°C至60°C)，请检查单位是否为摄氏度",
                UserWarning
            )
        
        if np.any(tmax_arr < tmin_arr):
            raise ValueError("最高气温不能低于最低气温")
        
        # 太阳辐射检查
        if np.any(rs_arr < 0) or np.any(rs_arr > 50):
            warnings.warn(
                "太阳辐射超出合理范围(0-50 MJ/m²/day)，请检查单位",
                UserWarning
            )
        
        # 相对湿度检查
        if rh_arr is not None:
            if np.any(rh_arr < 0) or np.any(rh_arr > 100):
                raise ValueError("相对湿度必须在0-100%之间")
        
        # 风速检查
        if uz_arr is not None:
            if np.any(uz_arr < 0) or np.any(uz_arr > 50):
                warnings.warn(
                    "风速超出合理范围(0-50 m/s)，请检查数据",
                    UserWarning
                )


# 辅助函数

def convert_rsds_to_mj(rsds: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    将ISIMIP太阳辐射单位转换为FAO-56所需单位
    
    参数
    ----
    rsds : float or Series
        太阳辐射 (W/m²)
    
    返回
    ----
    rs : float or Series
        太阳辐射 (MJ/m²/day)
    
    公式
    ----
    1 W/m² = 1 J/m²/s
    1 day = 86400 s
    1 MJ = 1e6 J
    
    因此: MJ/m²/day = W/m² × 86400 / 1e6 = W/m² × 0.0864
    """
    return rsds * 0.0864


def estimate_missing_tmax_tmin(
    tmean: pd.Series,
    trange: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """
    当缺少tmax/tmin时，基于tmean估算
    
    参数
    ----
    tmean : Series
        平均气温 (°C)
    trange : float
        假设的日温差 (°C), 默认10°C
    
    返回
    ----
    tmax, tmin : tuple of Series
        估算的最高/最低气温 (°C)
    
    警告
    ----
    这是一个粗略估计，仅用于数据缺失时的应急处理。
    建议使用实测数据或气候学日温差数据。
    """
    warnings.warn(
        "使用估算的tmax/tmin，可能影响PET计算精度。"
        "建议使用实测气温数据。",
        UserWarning
    )
    
    half_range = trange / 2
    tmax = tmean + half_range
    tmin = tmean - half_range
    
    return tmax, tmin


def validate_pet_reasonableness(
    pet: Union[float, pd.Series],
    p: Optional[Union[float, pd.Series]] = None
) -> Tuple[bool, str]:
    """
    检查PET计算结果的合理性
    
    参数
    ----
    pet : float or Series
        计算得到的PET (mm/year)
    p : float or Series, optional
        对应的降水量 (mm/year), 用于PET/P比值检查
    
    返回
    ----
    is_valid : bool
        是否合理
    message : str
        验证信息
    """
    pet_arr = np.asarray(pet)
    
    # 检查负值
    if np.any(pet_arr < 0):
        return False, "PET存在负值，计算可能有误"
    
    # 先检查干旱指数（如果提供了降水）
    if p is not None:
        p_arr = np.asarray(p)
        aridity_index = pet_arr / (p_arr + 1e-6)
        
        if np.any(aridity_index > 50):  # 放宽阈值，极端干旱地区可能PET/P很大
            return False, f"干旱指数(PET/P)过高: {aridity_index.max():.1f}"
    
    # 检查极端值
    if np.any(pet_arr > 5000):
        return False, "PET超过5000 mm/year，异常偏高"
    
    return True, "PET数值合理"


def calculate_aridity_classification(
    pet: float,
    p: float
) -> Tuple[float, str]:
    """
    基于干旱指数(PET/P)进行气候分类
    
    参数
    ----
    pet : float
        年均PET (mm)
    p : float
        年均降水 (mm)
    
    返回
    ----
    aridity_index : float
        干旱指数
    classification : str
        气候分类
    
    分类标准
    --------
    < 1.0 : 湿润
    1.0-2.0 : 半湿润
    2.0-5.0 : 半干旱
    > 5.0 : 干旱
    """
    ai = pet / (p + 1e-6)
    
    if ai < 1.0:
        climate = "湿润"
    elif ai < 2.0:
        climate = "半湿润"
    elif ai < 5.0:
        climate = "半干旱"
    else:
        climate = "干旱"
    
    return ai, climate
