"""
数据预处理模块
============

包含以下子模块:
- grdc_parser: GRDC观测径流数据解析
- climate_processor: ISIMIP气候数据处理
- pet_calculator: FAO-56 PET计算
- water_use_aggregator: 人类用水数据处理
"""

from .grdc_parser import GRDCParser

__all__ = ['GRDCParser']
