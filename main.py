"""
Budyko径流归因分析系统 - 主运行脚本

根据"代码撰写总体建议"第3.3节要求实现的主流程脚本。

功能：
1. 加载配置文件
2. 初始化日志系统
3. 数据预处理（GRDC、ISIMIP、Huang2018）
4. 执行Budyko归因分析
5. 生成结果报告
6. 异常处理和质量控制

使用示例：
    python main.py
    python main.py --config config/custom_config.yaml
    python main.py --stations 6435060,6243500
    python main.py --skip-validation

作者: Budyko归因分析系统开发团队
日期: 2025-01-01
"""

import yaml
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

# 导入项目模块
from src.data_preprocessing import GRDCParser, ClimateDataProcessor
from src.budyko_model import BudykoModel
from src.budyko_attribution import BudykoAttribution
from src.validation import QualityChecker


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    配置日志系统
    
    Parameters
    ----------
    log_level : str
        日志级别（DEBUG, INFO, WARNING, ERROR）
    log_file : str, optional
        日志文件路径（如果为None，只输出到控制台）
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件
    
    Parameters
    ----------
    config_path : str
        配置文件路径
        
    Returns
    -------
    config : dict
        配置字典
    """
    logger = logging.getLogger(__name__)
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    """
    验证配置文件的完整性
    
    Parameters
    ----------
    config : dict
        配置字典
        
    Returns
    -------
    is_valid : bool
        配置是否有效
    """
    logger = logging.getLogger(__name__)
    
    # 必需的配置项
    required_keys = ['data', 'parameters', 'quality_checks', 'output']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"配置文件缺少必需项: {key}")
            return False
    
    # 检查数据目录
    data_paths = config['data']
    for key, path in data_paths.items():
        if key.endswith('_dir'):
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"数据目录不存在（将自动创建）: {path}")
                path_obj.mkdir(parents=True, exist_ok=True)
    
    logger.info("配置文件验证通过")
    return True


def load_station_data(config: dict, station_ids: Optional[List[str]] = None) -> Dict:
    """
    加载GRDC站点数据
    
    Parameters
    ----------
    config : dict
        配置字典
    station_ids : list, optional
        指定加载的站点ID列表（如果为None，加载所有站点）
        
    Returns
    -------
    stations : dict
        站点数据字典，格式为 {station_id: DataFrame}
    """
    logger = logging.getLogger(__name__)
    logger.info("开始加载GRDC站点数据...")
    
    grdc_dir = Path(config['data']['grdc_dir'])
    parser = GRDCParser(grdc_dir)
    
    try:
        if station_ids:
            stations = {}
            for sid in station_ids:
                stations[sid] = parser.parse_station(sid)
            logger.info(f"成功加载 {len(stations)} 个指定站点")
        else:
            stations = parser.load_all_stations()
            logger.info(f"成功加载 {len(stations)} 个站点")
        
        return stations
    
    except Exception as e:
        logger.error(f"加载GRDC数据失败: {e}")
        return {}


def process_single_station(
    station_id: str,
    station_data: pd.DataFrame,
    config: dict,
    quality_checker: QualityChecker
) -> Optional[Dict]:
    """
    处理单个站点的归因分析
    
    Parameters
    ----------
    station_id : str
        站点ID
    station_data : DataFrame
        站点数据
    config : dict
        配置字典
    quality_checker : QualityChecker
        质量检查器实例
        
    Returns
    -------
    result : dict or None
        归因分析结果（如果失败返回None）
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 执行归因分析
        attribution = BudykoAttribution(station_data)
        attribution.set_periods(config['parameters']['change_year'])
        result = attribution.run_attribution()
        
        if result is None:
            logger.warning(f"站点 {station_id}: 归因分析失败（径流变化过小）")
            return None
        
        # 质量检查
        check_results = quality_checker.comprehensive_check(
            P=result.get('P_mean', 0),
            Q=result.get('Q_mean', 0),
            E=result.get('E_mean', 0),
            n=result.get('n_full', 0),
            eps_P=result['elasticity']['eps_P'],
            eps_PET=result['elasticity']['eps_PET'],
            eps_n=result['elasticity']['eps_n'],
            config=config['quality_checks']
        )
        
        # 记录检查结果
        all_passed = quality_checker.log_check_results(check_results, station_id)
        
        if not all_passed:
            logger.warning(f"站点 {station_id}: 部分质量检查未通过")
        
        # 添加站点ID到结果
        result['station_id'] = station_id
        result['quality_check_passed'] = all_passed
        
        return result
    
    except Exception as e:
        logger.error(f"站点 {station_id} 处理失败: {e}", exc_info=True)
        return None


def run_batch_analysis(
    stations: Dict,
    config: dict,
    skip_validation: bool = False
) -> pd.DataFrame:
    """
    批量处理多个站点
    
    Parameters
    ----------
    stations : dict
        站点数据字典
    config : dict
        配置字典
    skip_validation : bool
        是否跳过质量验证
        
    Returns
    -------
    results_df : DataFrame
        结果数据框
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始批量处理 {len(stations)} 个站点...")
    
    # 初始化质量检查器
    quality_checker = QualityChecker()
    
    results = []
    failed_stations = []
    
    # 使用进度条
    for station_id, station_data in tqdm(stations.items(), desc="处理站点"):
        logger.info(f"处理站点: {station_id}")
        
        result = process_single_station(
            station_id, station_data, config, quality_checker
        )
        
        if result is not None:
            results.append(result)
        else:
            failed_stations.append(station_id)
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info(f"批量处理完成!")
    logger.info(f"成功处理: {len(results)} 个站点")
    logger.info(f"失败站点: {len(failed_stations)} 个")
    
    if failed_stations:
        logger.warning(f"失败站点列表: {', '.join(failed_stations)}")
    
    # 转换为DataFrame
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    else:
        logger.error("所有站点处理失败，无结果输出")
        return pd.DataFrame()


def save_results(results_df: pd.DataFrame, config: dict):
    """
    保存结果到文件
    
    Parameters
    ----------
    results_df : DataFrame
        结果数据框
    config : dict
        配置字典
    """
    logger = logging.getLogger(__name__)
    
    if results_df.empty:
        logger.warning("结果为空，跳过保存")
        return
    
    # 创建输出目录
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存主结果文件
    output_file = output_dir / "attribution_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"主结果已保存: {output_file}")
    
    # 保存汇总统计
    summary_file = output_dir / "attribution_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Budyko径流归因分析 - 结果汇总\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"处理站点数: {len(results_df)}\n")
        f.write(f"质量检查通过: {results_df['quality_check_passed'].sum()}\n\n")
        
        # 贡献率统计
        f.write("贡献率平均值（%）:\n")
        f.write(f"  气候变化与变率（CCV）: {results_df['C_CCV'].mean():.2f} ± {results_df['C_CCV'].std():.2f}\n")
        f.write(f"  土地利用变化（LUCC）: {results_df['C_LUCC'].mean():.2f} ± {results_df['C_LUCC'].std():.2f}\n")
        f.write(f"  人类取用水（WADR）: {results_df['C_WADR'].mean():.2f} ± {results_df['C_WADR'].std():.2f}\n\n")
        
        # 参数统计
        f.write("流域参数n统计:\n")
        f.write(f"  平均值: {results_df['n_full'].mean():.3f}\n")
        f.write(f"  标准差: {results_df['n_full'].std():.3f}\n")
        f.write(f"  范围: [{results_df['n_full'].min():.3f}, {results_df['n_full'].max():.3f}]\n")
    
    logger.info(f"汇总统计已保存: {summary_file}")


def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Budyko径流归因分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                                    # 使用默认配置
  python main.py --config config/custom.yaml        # 指定配置文件
  python main.py --stations 6435060,6243500         # 仅处理指定站点
  python main.py --skip-validation                  # 跳过质量验证
  python main.py --log-file logs/run.log            # 指定日志文件
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='配置文件路径（默认: config/config.yaml）'
    )
    
    parser.add_argument(
        '--stations', '-s',
        help='指定站点ID列表，逗号分隔（如: 6435060,6243500）'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='跳过质量验证（不推荐）'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        help='日志文件路径（如果不指定，只输出到控制台）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 配置日志
    log_level = config['output'].get('log_level', 'INFO')
    setup_logging(log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Budyko径流归因分析系统启动")
    logger.info("="*60)
    
    # 验证配置
    if not validate_config(config):
        logger.error("配置文件验证失败，程序退出")
        sys.exit(1)
    
    # 解析站点ID列表
    station_ids = None
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(',')]
        logger.info(f"指定处理站点: {station_ids}")
    
    # 加载站点数据
    stations = load_station_data(config, station_ids)
    
    if not stations:
        logger.error("未能加载任何站点数据，程序退出")
        sys.exit(1)
    
    # 执行批量分析
    results_df = run_batch_analysis(
        stations, config, skip_validation=args.skip_validation
    )
    
    # 保存结果
    save_results(results_df, config)
    
    logger.info("="*60)
    logger.info("程序执行完成")
    logger.info("="*60)


if __name__ == "__main__":
    main()
