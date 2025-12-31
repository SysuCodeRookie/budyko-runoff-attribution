"""
ISIMIP气候数据处理模块
===================

功能:
- 读取ISIMIP NetCDF格式气候数据
- 根据流域边界进行空间提取和聚合
- 单位转换（符合Budyko方程要求）
- 时间聚合（日值→年值）
- 支持多变量处理（pr, tas, rsds, hurs, sfcWind等）

作者: [Your Name]
日期: 2025-12-31
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import warnings
from datetime import datetime

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    warnings.warn("rioxarray未安装，空间裁剪功能将受限。请运行: pip install rioxarray")


class ClimateDataProcessor:
    """
    ISIMIP气候数据处理器
    
    支持处理ISIMIP3a标准格式的NetCDF文件，包括：
    - GSWP3-W5E5大气强迫数据
    - 多种气候变量（降水、气温、辐射、湿度、风速等）
    
    Attributes:
        file_path (Path): NetCDF文件路径
        variable (str): 变量名称（pr, tas, rsds等）
        data (xr.Dataset or xr.DataArray): 加载的数据
        metadata (dict): 数据元信息
        
    Examples:
        >>> processor = ClimateDataProcessor("pr_GSWP3-W5E5_1960-2016.nc", variable="pr")
        >>> basin_shp = "yangtze_basin.shp"
        >>> basin_data = processor.extract_by_basin(basin_shp)
        >>> annual_pr = processor.aggregate_to_annual(basin_data)
    """
    
    # ISIMIP标准变量单位映射
    ISIMIP_UNITS = {
        'pr': 'kg m-2 s-1',          # 降水通量
        'tas': 'K',                   # 近地面气温
        'tasmax': 'K',                # 日最高气温
        'tasmin': 'K',                # 日最低气温
        'rsds': 'W m-2',              # 地表下行短波辐射
        'rlds': 'W m-2',              # 地表下行长波辐射
        'hurs': '%',                  # 相对湿度
        'sfcWind': 'm s-1',           # 近地面风速
        'ps': 'Pa',                   # 表面气压
        'dis': 'm3 s-1',              # 径流量
    }
    
    # 目标单位（Budyko方程所需）
    TARGET_UNITS = {
        'pr': 'mm year-1',            # 年降水量
        'tas': 'degC',                # 摄氏度
        'tasmax': 'degC',
        'tasmin': 'degC',
        'rsds': 'MJ m-2 day-1',       # FAO-56 PET计算所需
        'hurs': '%',                  # 保持不变
        'sfcWind': 'm s-1',           # 保持不变
        'ps': 'Pa',                   # 保持不变
        'dis': 'mm year-1',           # 径流深度
    }
    
    def __init__(self, file_path: Union[str, Path], variable: Optional[str] = None):
        """
        初始化气候数据处理器
        
        Args:
            file_path (str or Path): NetCDF文件路径，支持通配符（如"*.nc"）
            variable (str, optional): 要提取的变量名。如果为None，将自动检测
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists() and '*' not in str(file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        self.variable = variable
        self.data = None
        self.metadata = {}
        self._crs = "EPSG:4326"  # ISIMIP数据默认为WGS84
        
    def load_data(self, 
                  chunks: Optional[Dict] = None,
                  time_slice: Optional[slice] = None) -> xr.DataArray:
        """
        加载NetCDF数据
        
        Args:
            chunks (dict, optional): Dask分块参数，用于大文件延迟加载
                                     例如：{'time': 365, 'lat': 50, 'lon': 50}
            time_slice (slice, optional): 时间切片，例如：slice('1960', '2016')
            
        Returns:
            xr.DataArray: 加载的数据数组
            
        Examples:
            >>> processor.load_data(chunks={'time': 365})
            >>> processor.load_data(time_slice=slice('1980', '2000'))
        """
        # 支持多文件模式（使用通配符）
        if '*' in str(self.file_path):
            ds = xr.open_mfdataset(str(self.file_path), chunks=chunks, 
                                   combine='by_coords', parallel=True)
        else:
            ds = xr.open_dataset(self.file_path, chunks=chunks)
        
        # 自动检测变量名
        if self.variable is None:
            # 排除坐标变量
            coord_vars = {'time', 'lat', 'lon', 'latitude', 'longitude', 'x', 'y'}
            data_vars = set(ds.data_vars) - coord_vars
            if len(data_vars) == 1:
                self.variable = list(data_vars)[0]
            else:
                raise ValueError(
                    f"无法自动检测变量名。数据集包含多个变量: {data_vars}。"
                    f"请显式指定variable参数。"
                )
        
        # 提取指定变量
        if self.variable not in ds:
            raise ValueError(f"变量'{self.variable}'不在数据集中。可用变量: {list(ds.data_vars)}")
        
        data = ds[self.variable]
        
        # 时间切片
        if time_slice is not None:
            data = data.sel(time=time_slice)
        
        # 存储元信息
        self.metadata = {
            'variable': self.variable,
            'units': data.attrs.get('units', 'unknown'),
            'long_name': data.attrs.get('long_name', ''),
            'time_range': (str(data.time.min().values), str(data.time.max().values)),
            'spatial_resolution': self._get_resolution(data),
        }
        
        self.data = data
        return data
    
    def _get_resolution(self, data: xr.DataArray) -> Tuple[float, float]:
        """计算空间分辨率"""
        lat_name = 'lat' if 'lat' in data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in data.dims else 'longitude'
        
        if lat_name in data.coords and lon_name in data.coords:
            lat_res = abs(float(data[lat_name].diff(lat_name).mean()))
            lon_res = abs(float(data[lon_name].diff(lon_name).mean()))
            return (lat_res, lon_res)
        return (None, None)
    
    def extract_by_basin(self, 
                        basin_geometry: Union[str, Path, gpd.GeoDataFrame],
                        method: str = 'clip') -> xr.DataArray:
        """
        根据流域边界提取数据
        
        Args:
            basin_geometry (str, Path, or GeoDataFrame): 
                流域边界，可以是Shapefile路径或GeoDataFrame对象
            method (str): 提取方法
                - 'clip': 裁剪到流域范围（需要rioxarray）
                - 'bbox': 使用边界框提取（更快但不精确）
                - 'nearest': 最近邻点（单站点）
                
        Returns:
            xr.DataArray: 提取后的数据
            
        Notes:
            - 'clip'方法需要安装rioxarray
            - 对于不规则流域边界，建议使用'clip'方法
        """
        if self.data is None:
            self.load_data()
        
        # 加载流域几何
        if isinstance(basin_geometry, (str, Path)):
            basin_gdf = gpd.read_file(basin_geometry)
        elif isinstance(basin_geometry, gpd.GeoDataFrame):
            basin_gdf = basin_geometry
        else:
            raise TypeError("basin_geometry必须是Shapefile路径或GeoDataFrame对象")
        
        # 确保坐标系一致
        if basin_gdf.crs != self._crs:
            basin_gdf = basin_gdf.to_crs(self._crs)
        
        if method == 'clip':
            return self._clip_by_polygon(basin_gdf)
        elif method == 'bbox':
            return self._extract_by_bbox(basin_gdf)
        elif method == 'nearest':
            return self._extract_nearest_point(basin_gdf)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _clip_by_polygon(self, basin_gdf: gpd.GeoDataFrame) -> xr.DataArray:
        """使用多边形裁剪（精确但较慢）"""
        if not HAS_RIOXARRAY:
            raise ImportError("clip方法需要rioxarray库。请运行: pip install rioxarray")
        
        # 设置空间维度
        data_rio = self.data.rio.write_crs(self._crs)
        
        # 裁剪
        clipped = data_rio.rio.clip(
            basin_gdf.geometry.values,
            basin_gdf.crs,
            drop=True,
            all_touched=True  # 包含部分重叠的网格
        )
        
        return clipped
    
    def _extract_by_bbox(self, basin_gdf: gpd.GeoDataFrame) -> xr.DataArray:
        """使用边界框提取（快速但不精确）"""
        bounds = basin_gdf.total_bounds  # (minx, miny, maxx, maxy)
        
        lat_name = 'lat' if 'lat' in self.data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in self.data.dims else 'longitude'
        
        extracted = self.data.sel(
            {lon_name: slice(bounds[0], bounds[2]),
             lat_name: slice(bounds[3], bounds[1])}  # lat通常递减
        )
        
        return extracted
    
    def _extract_nearest_point(self, basin_gdf: gpd.GeoDataFrame) -> xr.DataArray:
        """提取最近邻点（用于站点数据）"""
        # 使用流域质心
        centroid = basin_gdf.geometry.centroid.iloc[0]
        
        lat_name = 'lat' if 'lat' in self.data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in self.data.dims else 'longitude'
        
        point_data = self.data.sel(
            {lon_name: centroid.x, lat_name: centroid.y},
            method='nearest'
        )
        
        return point_data
    
    def calculate_basin_mean(self, 
                            basin_data: xr.DataArray,
                            area_weighted: bool = True) -> xr.DataArray:
        """
        计算流域平均值
        
        Args:
            basin_data (xr.DataArray): 流域范围内的数据
            area_weighted (bool): 是否使用面积加权
            
        Returns:
            xr.DataArray: 流域平均时间序列
            
        Notes:
            - 面积加权考虑了纬度变化对网格面积的影响
            - 对于小流域，差异不大；对于大流域（跨纬度），建议使用面积加权
        """
        lat_name = 'lat' if 'lat' in basin_data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in basin_data.dims else 'longitude'
        
        if area_weighted and lat_name in basin_data.coords:
            # 计算纬度权重（cos(lat)）
            weights = np.cos(np.deg2rad(basin_data[lat_name]))
            weights.name = "weights"
            
            # 加权平均
            basin_mean = basin_data.weighted(weights).mean(dim=[lat_name, lon_name])
        else:
            # 简单平均
            spatial_dims = [d for d in [lat_name, lon_name] if d in basin_data.dims]
            basin_mean = basin_data.mean(dim=spatial_dims)
        
        return basin_mean
    
    def convert_units(self, 
                     data: Optional[xr.DataArray] = None,
                     target_unit: Optional[str] = None) -> xr.DataArray:
        """
        单位转换
        
        Args:
            data (xr.DataArray, optional): 要转换的数据。如果为None，使用self.data
            target_unit (str, optional): 目标单位。如果为None，使用预定义的目标单位
            
        Returns:
            xr.DataArray: 转换后的数据
            
        Examples:
            >>> # 降水: kg m-2 s-1 → mm/year
            >>> processor.convert_units()
            >>> # 温度: K → °C
            >>> processor.convert_units()
        """
        if data is None:
            if self.data is None:
                raise ValueError("未加载数据。请先调用load_data()")
            data = self.data
        
        variable = self.variable
        if variable not in self.ISIMIP_UNITS:
            warnings.warn(f"未知变量'{variable}'，跳过单位转换")
            return data
        
        # 确定目标单位
        if target_unit is None:
            target_unit = self.TARGET_UNITS.get(variable)
        
        # 执行转换
        converted = data.copy()
        
        if variable == 'pr':
            # kg m-2 s-1 → mm/year
            # 1 kg m-2 s-1 = 1 mm/s
            # mm/year = mm/s × 86400 s/day × 365.25 day/year
            converted = data * 86400 * 365.25
            converted.attrs['units'] = 'mm year-1'
            
        elif variable in ['tas', 'tasmax', 'tasmin']:
            # K → °C
            converted = data - 273.15
            converted.attrs['units'] = 'degC'
            
        elif variable == 'rsds':
            # W m-2 → MJ m-2 day-1
            # 1 W = 1 J/s
            # MJ m-2 day-1 = W m-2 × 86400 s/day / 1e6 J/MJ
            converted = data * 86400 / 1e6
            converted.attrs['units'] = 'MJ m-2 day-1'
            
        elif variable == 'dis':
            # 需要集水区面积进行转换
            warnings.warn(
                "径流量转换为径流深度需要集水区面积。"
                "请使用GRDCParser.convert_to_depth()方法。"
            )
        
        return converted
    
    def aggregate_to_annual(self, 
                           data: Optional[xr.DataArray] = None,
                           method: str = 'sum',
                           water_year: bool = False,
                           water_year_start_month: int = 10) -> pd.Series:
        """
        时间聚合为年值
        
        Args:
            data (xr.DataArray, optional): 要聚合的数据
            method (str): 聚合方法
                - 'sum': 年总和（用于降水、径流）
                - 'mean': 年平均（用于温度、湿度）
            water_year (bool): 是否使用水文年
            water_year_start_month (int): 水文年起始月份（1-12）
            
        Returns:
            pd.Series: 年值时间序列，索引为年份
            
        Examples:
            >>> annual_pr = processor.aggregate_to_annual(method='sum')
            >>> annual_tas = processor.aggregate_to_annual(method='mean')
        """
        if data is None:
            if self.data is None:
                raise ValueError("未加载数据")
            data = self.data
        
        # 转换为pandas时间序列（如果是多维，应已计算空间平均）
        if len(data.dims) > 1:
            warnings.warn(
                "数据包含多个维度，将先计算空间平均。"
                "建议先调用calculate_basin_mean()。"
            )
            spatial_dims = [d for d in data.dims if d != 'time']
            data = data.mean(dim=spatial_dims)
        
        # 转换为pandas Series
        ts = data.to_series()
        
        # 处理水文年
        if water_year:
            # 构建水文年标识
            df = pd.DataFrame({'value': ts})
            df['water_year'] = df.index.to_period(f'{water_year_start_month}M').apply(
                lambda x: x.year if x.month >= water_year_start_month else x.year
            )
            group_col = 'water_year'
        else:
            df = pd.DataFrame({'value': ts})
            df['year'] = df.index.year
            group_col = 'year'
        
        # 聚合
        if method == 'sum':
            annual = df.groupby(group_col)['value'].sum()
        elif method == 'mean':
            annual = df.groupby(group_col)['value'].mean()
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        annual.index.name = 'year'
        return annual
    
    def process_pipeline(self,
                        basin_geometry: Union[str, Path, gpd.GeoDataFrame],
                        convert_units: bool = True,
                        aggregate: bool = True,
                        aggregate_method: str = 'auto') -> pd.Series:
        """
        完整处理流水线
        
        一站式方法：加载→提取→转换→聚合
        
        Args:
            basin_geometry: 流域边界
            convert_units (bool): 是否进行单位转换
            aggregate (bool): 是否聚合为年值
            aggregate_method (str): 聚合方法（'auto'会根据变量自动选择）
            
        Returns:
            pd.Series: 处理后的年值时间序列
            
        Examples:
            >>> processor = ClimateDataProcessor("pr_data.nc", variable="pr")
            >>> annual_pr = processor.process_pipeline("basin.shp")
        """
        # 1. 加载数据
        print(f"加载数据: {self.file_path.name}")
        self.load_data()
        
        # 2. 流域提取
        print(f"提取流域数据...")
        basin_data = self.extract_by_basin(basin_geometry, method='clip')
        
        # 3. 空间平均
        print(f"计算流域平均...")
        basin_mean = self.calculate_basin_mean(basin_data, area_weighted=True)
        
        # 4. 单位转换
        if convert_units:
            print(f"单位转换: {self.metadata.get('units')} → {self.TARGET_UNITS.get(self.variable)}")
            basin_mean = self.convert_units(basin_mean)
        
        # 5. 时间聚合
        if aggregate:
            # 自动选择聚合方法
            if aggregate_method == 'auto':
                if self.variable in ['pr', 'dis']:
                    aggregate_method = 'sum'
                else:
                    aggregate_method = 'mean'
            
            print(f"聚合为年值（方法: {aggregate_method}）")
            result = self.aggregate_to_annual(basin_mean, method=aggregate_method)
        else:
            result = basin_mean
        
        print(f"✓ 处理完成")
        return result
    
    @staticmethod
    def batch_process_variables(file_pattern: str,
                                variables: List[str],
                                basin_geometry: Union[str, Path],
                                output_dir: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        批量处理多个变量
        
        Args:
            file_pattern (str): 文件路径模式（如"data/pr_*.nc"）
            variables (list): 变量名列表
            basin_geometry: 流域边界
            output_dir (str, optional): 输出目录（保存CSV）
            
        Returns:
            dict: {变量名: 年值序列}的字典
            
        Examples:
            >>> results = ClimateDataProcessor.batch_process_variables(
            ...     "data/*_GSWP3-W5E5.nc",
            ...     ['pr', 'tas', 'rsds'],
            ...     "yangtze_basin.shp"
            ... )
        """
        results = {}
        
        for var in variables:
            print(f"\n{'='*60}")
            print(f"处理变量: {var}")
            print(f"{'='*60}")
            
            try:
                # 构建文件路径（假设文件名包含变量名）
                file_path = str(file_pattern).replace('*', var)
                
                processor = ClimateDataProcessor(file_path, variable=var)
                annual_data = processor.process_pipeline(basin_geometry)
                
                results[var] = annual_data
                
                # 保存到文件
                if output_dir:
                    output_path = Path(output_dir) / f"{var}_annual.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    annual_data.to_csv(output_path, header=['value'])
                    print(f"已保存至: {output_path}")
                    
            except Exception as e:
                print(f"⚠️  处理变量'{var}'失败: {e}")
                continue
        
        return results


# 辅助函数
def validate_climate_data(P: float, PET: float, Q: float) -> Tuple[bool, str]:
    """
    验证气候数据的物理一致性
    
    Args:
        P (float): 降水量 (mm/year)
        PET (float): 潜在蒸散发 (mm/year)
        Q (float): 径流量 (mm/year)
        
    Returns:
        tuple: (is_valid, message)
    """
    # 检查非负性
    if P < 0 or PET < 0 or Q < 0:
        return False, f"存在负值: P={P:.1f}, PET={PET:.1f}, Q={Q:.1f}"
    
    # 检查水量平衡
    if Q > P:
        return False, f"径流({Q:.1f}) > 降水({P:.1f})，违反水量平衡"
    
    # 检查合理范围
    if P > 10000:  # 极端降水
        return True, f"降水量极高({P:.1f} mm/year)，请确认数据"
    
    if PET > 5000:  # 极端PET
        return True, f"PET极高({PET:.1f} mm/year)，请确认数据"
    
    return True, "通过检验"


def calculate_aridity_index(P: float, PET: float) -> Tuple[float, str]:
    """
    计算干旱指数（Aridity Index）
    
    AI = PET / P (干燥度指数)
    或 AI = P / PET (湿润度指数)
    
    Args:
        P (float): 降水量
        PET (float): 潜在蒸散发
        
    Returns:
        tuple: (aridity_index, climate_type)
    """
    if P <= 0 or PET <= 0:
        return np.nan, "数据无效"
    
    ai = PET / P  # 干燥度指数
    
    # 气候分类（UNESCO标准）
    if ai < 0.05:
        climate_type = "极端湿润"
    elif ai < 0.2:
        climate_type = "湿润"
    elif ai < 0.5:
        climate_type = "半湿润"
    elif ai < 0.65:
        climate_type = "半干旱"
    elif ai < 1.0:
        climate_type = "干旱"
    else:
        climate_type = "极端干旱"
    
    return ai, climate_type


if __name__ == "__main__":
    # 测试代码
    print("ISIMIP气候数据处理模块已加载")
    print("\n使用示例:")
    print("  processor = ClimateDataProcessor('pr_GSWP3-W5E5.nc', variable='pr')")
    print("  annual_pr = processor.process_pipeline('basin.shp')")
