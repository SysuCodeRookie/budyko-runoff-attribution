"""
GRDC数据解析模块
==============

功能:
- 解析GRDC文本格式观测径流数据
- 提取站点元数据（ID、坐标、集水区面积）
- 单位转换（m³/s → mm/year）
- 时间聚合（日值 → 年值）
- 数据质量过滤

作者: [Your Name]
日期: 2025-12-31
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings


class GRDCParser:
    """
    GRDC观测径流数据解析器
    
    支持GRDC标准文本格式（包含元数据头部和时间序列数据）。
    
    Attributes:
        file_path (Path): GRDC数据文件路径
        metadata (dict): 站点元数据
        timeseries (pd.DataFrame): 时间序列数据
    
    Examples:
        >>> parser = GRDCParser("data/raw/GRDC/6335020_Q_Day.Cmd.txt")
        >>> metadata = parser.parse_metadata()
        >>> print(f"站点ID: {metadata['grdc_no']}, 面积: {metadata['area_km2']} km²")
        >>> df_daily = parser.read_timeseries()
        >>> df_annual = parser.aggregate_to_annual()
        >>> df_depth = parser.convert_to_depth(metadata['area_km2'])
    """
    
    def __init__(self, file_path: str):
        """
        初始化GRDC解析器
        
        Args:
            file_path (str): GRDC数据文件的完整路径
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        self.metadata = {}
        self.timeseries = None
        self._raw_data = None
        
    def parse_metadata(self) -> Dict:
        """
        提取站点元数据
        
        从GRDC文件头部提取关键信息，包括:
        - GRDC站点编号
        - 站点名称
        - 河流名称
        - 经纬度坐标
        - 集水区面积
        - 海拔高度
        - 数据时间范围
        
        Returns:
            dict: 包含所有元数据的字典
            
        Examples:
            >>> parser = GRDCParser("station.txt")
            >>> meta = parser.parse_metadata()
            >>> print(meta['grdc_no'], meta['station'], meta['area_km2'])
        """
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 分离头部和数据部分
        data_start_marker = '# DATA'
        if data_start_marker in content:
            header, data_section = content.split(data_start_marker, 1)
        else:
            # 尝试其他可能的分隔符
            header = content.split('\n\n')[0]
        
        # 定义需要提取的字段及其正则表达式
        patterns = {
            'grdc_no': r'GRDC-No\.\s*:\s*(\d+)',
            'station': r'Station\s*:\s*(.+?)(?:\n|$)',
            'river': r'River\s*:\s*(.+?)(?:\n|$)',
            'country': r'Country\s*:\s*(.+?)(?:\n|$)',
            'latitude': r'Latitude \(DD\)\s*:\s*([-\d.]+)',
            'longitude': r'Longitude \(DD\)\s*:\s*([-\d.]+)',
            'area_km2': r'Catchment area \(km[²2]\)\s*:\s*([\d.]+)',
            'altitude_m': r'Altitude \(m ASL\)\s*:\s*([-\d.]+)',
            'time_series_start': r'First date\s*:\s*(\d{4}-\d{2}-\d{2})',
            'time_series_end': r'Last date\s*:\s*(\d{4}-\d{2}-\d{2})',
            'no_years': r'No. of years\s*:\s*([\d.]+)',
        }
        
        metadata = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, header)
            if match:
                value = match.group(1).strip()
                # 类型转换
                if key in ['grdc_no']:
                    metadata[key] = int(value)
                elif key in ['latitude', 'longitude', 'area_km2', 'altitude_m', 'no_years']:
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = None
                        warnings.warn(f"无法转换 {key} 为数值: {value}")
                else:
                    metadata[key] = value
            else:
                metadata[key] = None
                warnings.warn(f"未找到字段: {key}")
        
        self.metadata = metadata
        return metadata
    
    def read_timeseries(self, parse_dates: bool = True) -> pd.DataFrame:
        """
        读取时间序列数据
        
        从GRDC文件中读取径流观测数据（日值或月值）。
        数据列包括: YYYY-MM-DD, Value (m³/s), Flag
        
        Args:
            parse_dates (bool): 是否解析日期列为datetime对象，默认True
            
        Returns:
            pd.DataFrame: 包含列 ['date', 'discharge_m3s', 'flag']
            
        Notes:
            - 缺测值通常标记为 -999 或 NaN
            - Flag列标识数据质量（A=原始, C=计算, E=估计等）
        """
        # 读取整个文件
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 找到数据开始行
        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('# DATA'):
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError("未找到数据部分（# DATA标记）")
        
        # 跳过注释行，找到实际数据
        while data_start_idx < len(lines) and lines[data_start_idx].strip().startswith('#'):
            data_start_idx += 1
        
        # 读取数据（格式: YYYY-MM-DD;  流量;  标志）
        data_lines = lines[data_start_idx:]
        
        dates = []
        discharges = []
        flags = []
        
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 分割数据（可能是分号、空格或制表符分隔）
            parts = re.split(r'[;\s\t]+', line)
            if len(parts) >= 2:
                try:
                    date_str = parts[0]
                    discharge_str = parts[1]
                    flag_str = parts[2] if len(parts) >= 3 else 'U'  # U=Unknown
                    
                    # 处理缺测值
                    discharge = float(discharge_str)
                    if discharge < -900:  # -999通常表示缺测
                        discharge = np.nan
                    
                    dates.append(date_str)
                    discharges.append(discharge)
                    flags.append(flag_str)
                    
                except (ValueError, IndexError) as e:
                    warnings.warn(f"跳过无效行: {line} ({e})")
                    continue
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'discharge_m3s': discharges,
            'flag': flags
        })
        
        if parse_dates:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])  # 移除日期解析失败的行
            df = df.set_index('date').sort_index()
        
        self.timeseries = df
        return df
    
    def convert_to_depth(self, area_km2: Optional[float] = None) -> pd.DataFrame:
        """
        将径流从流量(m³/s)转换为水深(mm/year)
        
        转换公式:
            径流深度 (mm/year) = 流量 (m³/s) × 86400 (s/day) × 365.25 (day/year) 
                                 / [面积 (km²) × 10⁶ (m²/km²)] × 1000 (mm/m)
        
        Args:
            area_km2 (float, optional): 集水区面积(km²)。
                                        如果为None，从metadata中获取
            
        Returns:
            pd.DataFrame: 包含 'runoff_mm_year' 列的DataFrame
            
        Raises:
            ValueError: 如果area_km2未提供且metadata中也没有
            
        Examples:
            >>> df = parser.convert_to_depth()  # 自动使用metadata中的面积
            >>> df = parser.convert_to_depth(area_km2=1500)  # 手动指定面积
        """
        if self.timeseries is None:
            self.read_timeseries()
        
        if area_km2 is None:
            if 'area_km2' in self.metadata and self.metadata['area_km2']:
                area_km2 = self.metadata['area_km2']
            else:
                raise ValueError("集水区面积未知，请提供area_km2参数或确保metadata中有该信息")
        
        if area_km2 <= 0:
            raise ValueError(f"集水区面积必须为正值，当前值: {area_km2}")
        
        df = self.timeseries.copy()
        
        # 单位转换系数
        seconds_per_year = 86400 * 365.25
        area_m2 = area_km2 * 1e6
        mm_per_m = 1000
        
        # 转换公式
        df['runoff_mm_year'] = (df['discharge_m3s'] * seconds_per_year / area_m2) * mm_per_m
        
        return df
    
    def aggregate_to_annual(self, 
                           water_year: bool = False,
                           water_year_start_month: int = 10) -> pd.DataFrame:
        """
        将日值数据聚合为年值
        
        Args:
            water_year (bool): 是否使用水文年，默认False（日历年）
            water_year_start_month (int): 水文年起始月份（1-12），默认10（10月）
            
        Returns:
            pd.DataFrame: 年均值数据，包含列 ['year', 'discharge_m3s', 'data_count']
            
        Notes:
            - 缺测值（NaN）会被忽略
            - data_count列记录每年的有效观测天数
            
        Examples:
            >>> df_calendar = parser.aggregate_to_annual()  # 日历年
            >>> df_water = parser.aggregate_to_annual(water_year=True)  # 水文年
        """
        if self.timeseries is None:
            self.read_timeseries()
        
        df = self.timeseries.copy()
        
        if water_year:
            # 构建水文年标识
            # 例如: 2020-10-01 到 2021-09-30 标记为水文年2021
            df['water_year'] = df.index.to_period(f'{water_year_start_month}M').apply(
                lambda x: x.year if x.month >= water_year_start_month else x.year - 1
            )
            group_by_col = 'water_year'
        else:
            df['year'] = df.index.year
            group_by_col = 'year'
        
        # 年度聚合
        annual = df.groupby(group_by_col).agg({
            'discharge_m3s': ['mean', 'count'],
        })
        
        # 展平多层索引
        annual.columns = ['discharge_m3s', 'data_count']
        annual = annual.reset_index()
        annual = annual.rename(columns={group_by_col: 'year'})
        
        return annual
    
    def quality_filter(self, 
                       max_missing_pct: float = 15,
                       min_valid_days: int = 300) -> pd.DataFrame:
        """
        数据质量过滤
        
        过滤掉缺测率过高的年份，确保归因分析的可靠性。
        
        Args:
            max_missing_pct (float): 允许的最大缺测百分比（0-100），默认15%
            min_valid_days (int): 每年最少有效观测天数，默认300天
            
        Returns:
            pd.DataFrame: 过滤后的年值数据
            
        Notes:
            - 会同时检查缺测率和绝对观测天数两个条件
            - 被过滤的年份会在日志中记录
        """
        if self.timeseries is None:
            self.read_timeseries()
        
        # 获取年度数据
        annual = self.aggregate_to_annual()
        
        # 计算理论天数（考虑闰年）
        annual['expected_days'] = annual['year'].apply(
            lambda y: 366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365
        )
        
        # 计算缺测率
        annual['missing_pct'] = (1 - annual['data_count'] / annual['expected_days']) * 100
        
        # 应用过滤条件
        filtered = annual[
            (annual['missing_pct'] <= max_missing_pct) & 
            (annual['data_count'] >= min_valid_days)
        ].copy()
        
        # 报告过滤结果
        removed_count = len(annual) - len(filtered)
        if removed_count > 0:
            removed_years = set(annual['year']) - set(filtered['year'])
            warnings.warn(
                f"移除了 {removed_count} 个缺测率过高的年份: "
                f"{sorted(removed_years)}"
            )
        
        return filtered[['year', 'discharge_m3s']]
    
    def get_full_timeseries(self, 
                           convert_to_depth: bool = False,
                           area_km2: Optional[float] = None) -> pd.DataFrame:
        """
        获取完整的处理后时间序列
        
        一站式方法：解析元数据、读取数据、可选单位转换
        
        Args:
            convert_to_depth (bool): 是否转换为径流深度(mm/year)
            area_km2 (float, optional): 集水区面积，仅在convert_to_depth=True时需要
            
        Returns:
            pd.DataFrame: 处理后的完整时间序列
        """
        # 解析元数据
        if not self.metadata:
            self.parse_metadata()
        
        # 读取时间序列
        if self.timeseries is None:
            self.read_timeseries()
        
        # 单位转换
        if convert_to_depth:
            df = self.convert_to_depth(area_km2)
        else:
            df = self.timeseries.copy()
        
        return df
    
    @staticmethod
    def load_multiple_stations(grdc_dir: str, 
                              pattern: str = "*_Q_Day.Cmd.txt") -> Dict[int, 'GRDCParser']:
        """
        批量加载多个GRDC站点
        
        Args:
            grdc_dir (str): GRDC数据文件夹路径
            pattern (str): 文件名匹配模式，默认 "*_Q_Day.Cmd.txt"
            
        Returns:
            dict: {grdc_no: GRDCParser实例} 的字典
            
        Examples:
            >>> stations = GRDCParser.load_multiple_stations("data/raw/GRDC/")
            >>> for grdc_no, parser in stations.items():
            ...     meta = parser.parse_metadata()
            ...     print(f"站点 {grdc_no}: {meta['station']}")
        """
        grdc_path = Path(grdc_dir)
        if not grdc_path.exists():
            raise FileNotFoundError(f"目录不存在: {grdc_dir}")
        
        files = list(grdc_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"未找到匹配 '{pattern}' 的文件")
        
        stations = {}
        for file in files:
            try:
                parser = GRDCParser(str(file))
                metadata = parser.parse_metadata()
                grdc_no = metadata.get('grdc_no')
                
                if grdc_no:
                    stations[grdc_no] = parser
                else:
                    warnings.warn(f"文件 {file.name} 缺少GRDC编号，跳过")
                    
            except Exception as e:
                warnings.warn(f"加载文件 {file.name} 失败: {e}")
                continue
        
        print(f"成功加载 {len(stations)} 个GRDC站点")
        return stations


# 辅助函数
def validate_runoff_ratio(runoff_mm: float, precip_mm: float) -> Tuple[bool, str]:
    """
    验证径流系数的合理性
    
    Args:
        runoff_mm (float): 径流深度 (mm/year)
        precip_mm (float): 降水量 (mm/year)
        
    Returns:
        tuple: (is_valid, message)
    """
    if precip_mm <= 0:
        return False, "降水量必须为正值"
    
    ratio = runoff_mm / precip_mm
    
    if ratio < 0:
        return False, f"径流系数为负({ratio:.3f})，数据异常"
    elif ratio > 1.0:
        return False, f"径流系数>1({ratio:.3f})，违反水量平衡"
    elif ratio < 0.01:
        return True, f"径流系数极低({ratio:.3f})，可能为极端干旱区"
    elif ratio > 0.95:
        return True, f"径流系数极高({ratio:.3f})，请检查数据质量"
    else:
        return True, f"径流系数正常({ratio:.3f})"


if __name__ == "__main__":
    # 测试代码
    print("GRDC解析器模块已加载")
    print("使用示例:")
    print("  parser = GRDCParser('path/to/grdc_file.txt')")
    print("  metadata = parser.parse_metadata()")
    print("  annual_data = parser.aggregate_to_annual()")
