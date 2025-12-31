"""
validation模块：物理一致性检验和质量控制

本模块提供数据质量检查、物理约束验证、模型诊断等功能。
"""

from .quality_checks import QualityChecker

__all__ = ['QualityChecker']
