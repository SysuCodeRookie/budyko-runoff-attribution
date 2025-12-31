# Budyko径流归因分析系统

## 项目概述

基于Budyko理论的径流变化归因分析框架，用于量化气候变化、土地利用变化和人类取水活动对径流变化的贡献。

**适用领域**: 水文学、土木工程、水资源管理

## 项目结构

```
猫猫工作区/
├── src/                          # 源代码
│   ├── data_preprocessing/       # 数据预处理
│   │   ├── grdc_parser.py       # ✅ GRDC数据解析（已完成）
│   │   └── climate_processor.py # ✅ 气候数据处理（已完成）
│   ├── budyko_model/            # Budyko模型核心
│   │   ├── pet_calculator.py    # ✅ PET计算器（已完成）
│   │   ├── core_equations.py    # ✅ Budyko核心方程（已完成）
│   │   ├── parameter_calibration.py # ✅ 参数校准（已完成）
│   │   ├── elasticity_solver.py # ✅ 弹性系数求解（已完成）
│   │   ├── budyko_attribution.py # ✅ Budyko归因分析（已完成）
│   │   └── isimip_attribution.py # ✅ ISIMIP归因分析（已完成）
│   ├── validation/              # ✅ 质量检验（已完成）
│   │   └── quality_checks.py    # ✅ 物理一致性检查（已完成）
│   └── visualization/           # 结果可视化（待开发）
├── tests/                        # 单元测试
│   ├── test_grdc_parser.py      # ✅ GRDC测试（11/11通过）
│   ├── test_climate_processor.py # ✅ 气候处理测试（16/16通过）
│   ├── test_pet_calculator.py   # ✅ PET测试（27/27通过）
│   ├── test_core_equations.py   # ✅ 核心方程测试（46/46通过）
│   ├── test_parameter_calibration.py # ✅ 参数校准测试（30/30通过）
│   ├── test_elasticity_solver.py # ✅ 弹性系数测试（28/28通过）
│   ├── test_budyko_attribution.py # ✅ Budyko归因测试（25/25通过）
│   ├── test_isimip_attribution.py # ✅ ISIMIP归因测试（31/31通过）
│   └── test_quality_checks.py   # ✅ 质量检查测试（33/33通过）
├── examples/                     # 使用示例
│   ├── grdc_parser_example.py   # ✅ GRDC示例（已完成）
│   ├── climate_processor_example.py # ✅ 气候处理示例（已完成）
│   ├── pet_calculator_example.py # ✅ PET计算示例（已完成）
│   ├── core_equations_example.py # ✅ 核心方程示例（已完成）
│   ├── parameter_calibration_example.py # ✅ 参数校准示例（已完成）
│   ├── elasticity_solver_example.py # ✅ 弹性系数示例（已完成）
│   └── isimip_attribution_example.py # ✅ ISIMIP归因示例（已完成）
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后数据
│   └── results/                  # 结果数据
├── config/                       # 配置文件
└── notebooks/                    # Jupyter笔记本
```

## 快速开始

### 安装依赖

**推荐使用cat虚拟环境**（已配置并验证）:

```bash
# 激活cat环境
conda activate cat

# 环境信息
# Python: 3.11.14
# 已安装所有依赖包
# 测试状态: 128/128 通过 ✅ (模块1+2+3+4+5)
```

或从头创建环境:

```bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate budyko-env

# 或使用pip
pip install -r requirements.txt
```

### 基本使用

#### 示例1: 解析单个GRDC站点

```python
from src.data_preprocessing.grdc_parser import GRDCParser

# 初始化解析器
parser = GRDCParser("data/raw/GRDC/6335020_Q_Day.Cmd.txt")

# 提取元数据
metadata = parser.parse_metadata()
print(f"站点: {metadata['station']}, 面积: {metadata['area_km2']} km²")

# 读取时间序列并转换为年值
df_annual = parser.aggregate_to_annual()
print(df_annual.head())

# 转换为径流深度 (mm/year)
df_depth = parser.convert_to_depth()
```

#### 示例2: 批量站点参数校准和归因分析

```python
from src.budyko_model.parameter_calibration import ParameterCalibrator
import pandas as pd

# 初始化参数校准器
calibrator = ParameterCalibrator(change_point=1986)

# 创建多站点数据
stations_data = pd.DataFrame({
    'station_id': ['ST001', 'ST002', 'ST003'],
    'P': [850.0, 1200.0, 600.0],
    'PET': [1200.0, 900.0, 1400.0],
    'Q_n': [200.0, 500.0, 100.0]
})

# 批量校准
results = calibrator.batch_calibrate_stations(stations_data)
for r in results:
    print(f"站点 {r.station_id}: n={r.n:.3f}, 干旱指数={r.aridity_index:.2f}")
```

#### 示例3: 计算潜在蒸散发 (PET)

```python
from src.budyko_model.pet_calculator import PETCalculator

# 初始化PET计算器
calculator = PETCalculator(latitude=30.5, elevation=500)

# 单日计算（FAO-56 Penman-Monteith方法）
pet = calculator.calculate_fao56(
    tmean=25.0,    # 平均气温 (°C)
    tmax=30.0,     # 最高气温 (°C)
    tmin=20.0,     # 最低气温 (°C)
    rs=20.0,       # 太阳辐射 (MJ m⁻² day⁻¹)
    rh=60.0,       # 相对湿度 (%)
    uz=2.0         # 2m风速 (m/s)
)
print(f"日PET: {pet:.2f} mm/day")

# 年度时间序列计算
df_climate = pd.read_csv("data/processed/climate_data.csv")
pet_annual = calculator.aggregate_to_annual(
    calculator.calculate_fao56(
        tmean=df_climate['tas'],
        tmax=df_climate['tasmax'],
        tmin=df_climate['tasmin'],
        rs=df_climate['rsds'],
        rh=df_climate['hurs'],
        uz=df_climate['sfcWind']
    ),
    dates=df_climate['time']
)
print(f"年均PET: {pet_annual.mean():.2f} mm/year")
```

#### 示例3: 处理ISIMIP气候数据

```python
from src.data_preprocessing.climate_processor import ClimateDataProcessor

# 初始化处理器
processor = ClimateDataProcessor(
    "data/raw/ISIMIP/pr_GSWP3-W5E5_1960-2016.nc",
    variable="pr"
)

# 完整流水线处理
annual_pr = processor.process_pipeline(
    basin_geometry="data/shapefiles/yangtze_basin.shp",
    convert_units=True,        # kg m⁻² s⁻¹ → mm/year
    aggregate=True,            # 日值 → 年值
    aggregate_method='sum'     # 降水用求和
)

print(f"年均降水: {annual_pr.mean():.2f} mm/year")
```

#### 示例5: 批量处理多个气候变量

```python
# 批量处理
results = ClimateDataProcessor.batch_process_variables(
    file_pattern="data/raw/ISIMIP/{var}_GSWP3-W5E5_1960-2016.nc",
    variables=['pr', 'tas', 'rsds'],
    basin_geometry="data/shapefiles/basin.shp",
    output_dir="data/processed/climate/"
)

# 访问结果
P = results['pr']    # 年降水量 (mm/year)
T = results['tas']   # 年均气温 (°C)
R = results['rsds']  # 太阳辐射 (MJ m⁻² day⁻¹)
```

## 已完成模块

### ✅ 模块1: GRDC数据解析器 (grdc_parser.py)

**功能**:
- 解析GRDC标准文本格式
- 提取站点元数据（ID、坐标、集水区面积等）
- 时间序列读取（日值）
- 单位转换：m³/s → mm/year
- 时间聚合：日值 → 年值
- 数据质量过滤（缺测率控制）
- 批量站点加载

**关键方法**:
| 方法 | 功能 |
|------|------|
| `parse_metadata()` | 提取站点元数据 |
| `read_timeseries()` | 读取时间序列 |
| `convert_to_depth()` | 流量→水深转换 |
| `aggregate_to_annual()` | 日值→年值聚合 |
| `quality_filter()` | 质量过滤 |
| `load_multiple_stations()` | 批量加载（静态方法） |

**测试覆盖率**: ✅ 已完成（`tests/test_grdc_parser.py`）

**使用示例**: ✅ 已提供（`examples/grdc_parser_example.py`）

### ✅ 模块2: 气候数据处理器 (climate_processor.py)

**功能**:
- 读取ISIMIP NetCDF格式气候数据
- 支持三种流域提取方法（clip/bbox/nearest）
- 面积加权平均（考虑纬度变化）
- 自动单位转换（pr: kg m⁻² s⁻¹ → mm/year, tas: K → °C, rsds: W m⁻² → MJ m⁻² day⁻¹）
- 时间聚合（日值→年值，支持水文年）
- 完整处理流水线（process_pipeline）
- 批量变量处理（batch_process_variables）
- 数据质量检查（水量平衡验证）
- 干旱指数计算与气候分类

**关键方法**:
| 方法 | 功能 |
|------|------|
| `load_data()` | 加载NetCDF数据（支持chunks延迟加载） |
| `extract_by_basin()` | 流域空间提取（clip/bbox/nearest） |
| `calculate_basin_mean()` | 流域平均（面积加权） |
| `convert_units()` | 自动单位转换 |
| `aggregate_to_annual()` | 时间聚合（支持水文年） |
| `process_pipeline()` | 完整处理流水线 |
| `batch_process_variables()` | 批量处理（静态方法） |

**辅助函数**:
| 函数 | 功能 |
|------|------|
| `validate_climate_data()` | 水量平衡检查 |
| `calculate_aridity_index()` | 干旱指数与气候分类 |

**测试覆盖率**: ✅ 已完成（`tests/test_climate_processor.py`, 16/16通过）

**使用示例**: ✅ 已提供（`examples/climate_processor_example.py`, 7个场景）

### ✅ 模块3: PET计算器 (pet_calculator.py)

**功能**:
- FAO-56 Penman-Monteith标准方法（基于pyet库）
- Hargreaves简化方法（仅需温度数据）
- 与ISIMIP气候数据无缝集成
- 自动单位转换（rsds: W m⁻² → MJ m⁻² day⁻¹）
- 时间聚合（日值→年值，支持sum/mean）
- 数据质量检查（温度范围、湿度边界、PET合理性）
- 干旱指数计算与气候分类（联合PET/P）
- 缺测值估算（tmax/tmin从tas推算）

**关键方法**:
| 方法 | 功能 |
|------|------|
| `calculate_fao56()` | FAO-56 Penman-Monteith法（标准） |
| `calculate_hargreaves()` | Hargreaves简化法（仅需温度） |
| `calculate_from_climate_data()` | 从ClimateDataProcessor输出计算 |
| `aggregate_to_annual()` | 日值→年值聚合 |

**辅助函数**:
| 函数 | 功能 |
|------|------|
| `convert_rsds_to_mj()` | ISIMIP辐射单位转换 |
| `estimate_missing_tmax_tmin()` | 从平均气温估算极值 |
| `validate_pet_reasonableness()` | PET合理性检查 |
| `calculate_aridity_classification()` | 气候类型分类（基于PET/P） |

**测试覆盖率**: ✅ 已完成（`tests/test_pet_calculator.py`, 27/27通过）

**使用示例**: ✅ 已提供（`examples/pet_calculator_example.py`, 8个场景）

### ✅ 模块4: Budyko核心方程 (core_equations.py)

**功能**:
- Choudhury-Yang参数化Budyko方程实现
- 实际蒸散发（E）和天然径流（Q_n）计算
- 参数n反演（模型校准）- 基于scipy.optimize
- 径流弹性系数计算（εP, εPET, εn）
- 归因分解（CCV和LUCC贡献）
- 物理一致性检查（水量平衡验证）
- 干旱指数计算与气候分类

**核心方程**:
```python
# Budyko方程
E = (P × PET) / (P^n + PET^n)^(1/n)
Q_n = P - E

# 弹性系数（main.tex方程4-6）
εP, εPET, εn = f(P, PET, n)

# 归因分解（main.tex方程7）
ΔQ_n = εP × (Q_n/P) × ΔP + εPET × (Q_n/PET) × ΔPET + εn × (Q_n/n) × Δn
```

**关键方法**:
| 方法 | 功能 |
|------|------|
| `calculate_actual_ET()` | 计算实际蒸散发E |
| `calculate_naturalized_runoff()` | 计算天然径流Q_n |
| `calibrate_parameter_n()` | 反演参数n（Brent/Newton方法） |
| `calculate_elasticities()` | 计算三个弹性系数 |
| `calculate_runoff_change_attribution()` | 完整归因分析（main.tex Step 1-4） |

**辅助函数**:
| 函数 | 功能 |
|------|------|
| `validate_water_balance()` | 检查Q_n < P约束 |
| `calculate_aridity_index()` | 计算干旱指数φ = PET/P |
| `estimate_n_from_climate()` | 初始n值估算 |

**测试覆盖率**: ✅ 已完成（`tests/test_core_equations.py`, 44/44通过）

**使用示例**: ✅ 已提供（`examples/core_equations_example.py`, 7个场景）

### ✅ 模块5: 参数校准和归因分析 (parameter_calibration.py)

**功能**:
- 批量站点参数n校准（支持并行处理）
- 时段划分和参数演变分析（检测LUCC信号）
- 完整归因分解（CCV、LUCC、WADR三因子）
- Bootstrap不确定性估计（置信区间计算）
- 数据质量验证和异常检测
- 区域集合归因统计（多站点综合）
- 结果导出（CSV格式）

**核心类**:
```python
ParameterCalibrator:
    - calibrate_single_station()        # 单站点校准
    - batch_calibrate_stations()        # 批量站点处理
    - analyze_parameter_evolution()     # 时段对比分析
    - calculate_attribution()           # 归因分解
    - bootstrap_uncertainty()           # 不确定性评估
    - export_results()                  # 结果导出

CalibrationResult:   # 校准结果数据容器
AttributionResult:   # 归因结果数据容器
```

**归因分解公式（main.tex方程8-10）**:
```python
# 气候变化贡献
C_CCV = (ΔQ_CCV / ΔQ_obs) × 100%
      = [(εP × Q_n/P × ΔP + εPET × Q_n/PET × ΔPET) / ΔQ_obs] × 100%

# 土地利用变化贡献
C_LUCC = (ΔQ_LUCC / ΔQ_obs) × 100%
       = [(εn × Q_n/n × Δn) / ΔQ_obs] × 100%

# 人类取用水贡献
C_WADR = [(ΔQ_obs - ΔQ_n) / ΔQ_obs] × 100%
```

**关键方法**:
| 方法 | 功能 |
|------|------|
| `calibrate_single_station()` | 单站点参数校准（包含物理检查） |
| `batch_calibrate_stations()` | 批量站点处理（支持并行） |
| `analyze_parameter_evolution()` | 时段划分（如1960-1985 vs 1986-2016） |
| `calculate_attribution()` | 完整归因分解（CCV+LUCC+WADR） |
| `bootstrap_uncertainty()` | Bootstrap重采样不确定性 |

**辅助函数**:
| 函数 | 功能 |
|------|------|
| `validate_time_series_quality()` | 数据质量检查（缺测、负值） |
| `calculate_ensemble_attribution()` | 区域多站点集合统计 |

**测试覆盖率**: ✅ 已完成（`tests/test_parameter_calibration.py`, 30/30通过）

**使用示例**: ✅ 已提供（`examples/parameter_calibration_example.py`, 7个场景）

### ✅ 模块6: 弹性系数求解器 (elasticity_solver.py)

**功能**:
- 高性能弹性系数计算（向量化实现）
- 支持三种弹性系数（εP, εPET, εn）
- 干旱指数计算与气候分类
- 水量平衡验证
- 弹性系数合理性检查
- 批量时间序列处理
- 物理约束验证（符号检查、数值范围）

**核心方程（main.tex方程4-6）**:
```python
εP = [1 - ((φⁿ)/(1+φⁿ))^(1/n+1)] / [1 - ((φⁿ)/(1+φⁿ))^(1/n)]

εPET = [1/(1+φⁿ)] × [1/(1-((1+φⁿ)/φⁿ)^(1/n))]

εn = 1/[(1+φⁿ)^(1/n)-1] × [Pⁿln(P)+PETⁿln(PET))/(Pⁿ+PETⁿ) - ln(Pⁿ+PETⁿ)/n]
```

**关键方法**:
| 方法 | 功能 |
|------|------|
| `calculate_elasticity_P()` | 降水弹性系数（εP > 0） |
| `calculate_elasticity_PET()` | PET弹性系数（εPET < 0） |
| `calculate_elasticity_n()` | 参数n弹性系数（εn < 0） |
| `calculate_all_elasticities()` | 一次性计算所有弹性系数 |
| `process_timeseries()` | 批量时间序列处理 |

**辅助函数**:
| 函数 | 功能 |
|------|------|
| `validate_elasticity_signs()` | 符号物理意义检查 |
| `calculate_aridity_index()` | 干旱指数φ = PET/P |

**测试覆盖率**: ✅ 已完成（`tests/test_elasticity_solver.py`, 28/28通过）

**使用示例**: ✅ 已提供（`examples/elasticity_solver_example.py`, 6个场景）

### ✅ 模块7: Budyko归因分析 (budyko_attribution.py)

**功能**:
- 完整Budyko归因分析流程（main.tex Step 1-6）
- 时段划分与对比分析（基准期vs变化期）
- 三因子归因分解（CCV、LUCC、WADR）
- 参数演变分析（检测LUCC信号）
- 数据质量验证（水量平衡、参数范围）
- 批量站点处理
- 结果汇总与导出（CSV格式）

**归因方程（main.tex方程7-10）**:
```python
# 天然径流变化分解
ΔQ̂_n = εP × (Q_n/P) × ΔP + εPET × (Q_n/PET) × ΔPET + εn × (Q_n/n) × Δn
       ⎣━━━━━━ ΔQ_n,CCV ━━━━━━⎦   ⎣━━━━ ΔQ_n,LUCC ━━━━⎦

# 贡献率计算
C_CCV = (ΔQ_n,CCV / ΔQ_o) × 100%   # 气候变化贡献
C_LUCC = (ΔQ_n,LUCC / ΔQ_o) × 100%  # 土地利用变化贡献
C_WADR = [(ΔQ_o - ΔQ_n) / ΔQ_o] × 100%  # 人类取用水贡献
```

**核心类**:
```python
BudykoAttribution:
    - set_periods()                 # 时段划分（如1960-1985 vs 1986-2016）
    - run_attribution()             # 完整归因流程（main.tex 6步骤）
    - calculate_parameter_evolution() # 参数n时间演变
    - validate_results()            # 结果物理合理性检查

AttributionResult:  # 归因结果数据容器
```

**关键方法**:
| 方法 | 功能 |
|------|------|
| `set_periods()` | 设置基准期和变化期 |
| `run_attribution()` | 执行完整归因分析（6步骤） |
| `calculate_contributions()` | 计算三因子贡献率 |
| `batch_process_stations()` | 批量站点处理 |
| `export_results()` | 导出结果到CSV |

**测试覆盖率**: ✅ 已完成（`tests/test_budyko_attribution.py`, 25/25通过）

**使用示例**: ✅ 已提供（`examples/budyko_attribution_example.py` 在parameter_calibration中）

### ✅ 模块8: ISIMIP归因分析 (isimip_attribution.py)

**功能**:
- ISIMIP3a模型数据集成与处理
- 多情景对比分析（obsclim、counterclim、histsoc、1901soc）
- ACC与NCV分离（人为气候变化 vs 自然变率）
- 多模型集合统计（9个GHMs）
- 模型不确定性量化（标准差、变异系数）
- 与Budyko方法结果对比验证

**ISIMIP情景定义（main.tex）**:
```python
# 模型输出情景
Q'_o   = obsclim + histsoc     # 观测气候 + 历史人类影响
Q'_n   = obsclim + 1901soc     # 观测气候 + 1901年固定人类影响
Q'_cn  = counterclim + 1901soc # 去趋势气候 + 1901年固定人类影响

# 归因分解
C_CCV  = ΔQ'_n / ΔQ_o × 100%              # 总气候效应
C_ACC  = (ΔQ'_n - ΔQ'_cn) / ΔQ_o × 100%  # 人为气候变化
C_NCV  = ΔQ'_cn / ΔQ_o × 100%             # 自然气候变率
C_LUCC = (ΔQ_n - ΔQ'_n) / ΔQ_o × 100%    # 土地利用变化
C_WADR = (ΔQ_o - ΔQ_n) / ΔQ_o × 100%     # 人类取用水
```

**核心类**:
```python
ISIMIPAttribution:
    - load_model_outputs()          # 加载ISIMIP模型输出
    - calculate_contributions()     # 计算ACC/NCV分离的贡献
    - ensemble_statistics()         # 多模型集合统计
    - compare_with_budyko()         # 与Budyko方法对比

ISIMIPResult:  # ISIMIP归因结果容器
```

**关键方法**:
| 方法 | 功能 |
|------|------|
| `load_model_outputs()` | 读取ISIMIP NetCDF输出 |
| `calculate_acc_ncv()` | 分离ACC和NCV贡献 |
| `ensemble_mean()` | 多模型集合平均 |
| `calculate_uncertainty()` | 模型间不确定性 |
| `export_results()` | 导出完整归因结果 |

**测试覆盖率**: ✅ 已完成（`tests/test_isimip_attribution.py`, 31/31通过）

**使用示例**: ✅ 已提供（`examples/isimip_attribution_example.py`, 6个场景）

## 待开发模块

### 🔲 模块9: 结果可视化
- 归因结果图表生成（柱状图、饼图、瀑布图）
- Budyko空间轨迹图
- 时间序列变化趋势图
- 多站点对比可视化
- 交互式Dashboard（基于Plotly/Dash）

### 🔲 模块10: 报告生成
- 自动化报告生成（PDF/HTML）
- 多站点批量报告
- 自定义报告模板

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_grdc_parser.py -v           # 模块1测试（11个）
pytest tests/test_climate_processor.py -v     # 模块2测试（16个）
pytest tests/test_pet_calculator.py -v        # 模块3测试（27个）
pytest tests/test_core_equations.py -v        # 模块4测试（46个）
pytest tests/test_parameter_calibration.py -v # 模块5测试（30个）
pytest tests/test_elasticity_solver.py -v     # 模块6测试（28个）
pytest tests/test_budyko_attribution.py -v    # 模块7测试（25个）
pytest tests/test_isimip_attribution.py -v    # 模块8测试（31个）
pytest tests/test_quality_checks.py -v        # 质量检查测试（33个）

# 查看测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 运行示例

```bash
# 运行GRDC解析器示例
python examples/grdc_parser_example.py

# 运行气候数据处理示例
python examples/climate_processor_example.py

# 运行PET计算示例
python examples/pet_calculator_example.py

# 运行Budyko核心方程示例
python examples/core_equations_example.py

# 运行参数校准和归因分析示例
python examples/parameter_calibration_example.py

# 运行弹性系数求解示例
python examples/elasticity_solver_example.py

# 运行ISIMIP归因分析示例
python examples/isimip_attribution_example.py
```

## 开发进度

| 阶段 | 模块 | 状态 | 完成日期 |
|------|------|------|----------|
| 阶段一 | grdc_parser.py | ✅ 完成 | 2025-12-31 |
| 阶段一 | climate_processor.py | ✅ 完成 | 2025-12-31 |
| 阶段一 | pet_calculator.py | ✅ 完成 | 2025-01-01 |
| 阶段二 | core_equations.py | ✅ 完成 | 2025-01-01 |
| 阶段二 | parameter_calibration.py | ✅ 完成 | 2025-01-01 |
| 阶段二 | elasticity_solver.py | ✅ 完成 | 2025-01-01 |
| 阶段三 | budyko_attribution.py | ✅ 完成 | 2025-01-01 |
| 阶段三 | isimip_attribution.py | ✅ 完成 | 2025-01-01 |
| 阶段三 | quality_checks.py | ✅ 完成 | 2025-01-01 |
| 阶段四 | 结果可视化 | 🔲 待开发 | - |
| 阶段四 | 报告生成 | 🔲 待开发 | - |

**整体进度**: 8/10 核心模块完成 (80%)
**当前版本**: v0.8.0-beta
**测试状态**: 247/247 通过 ✅

## 数据要求

### GRDC数据
- 格式: 标准GRDC文本文件 (`*_Q_Day.Cmd.txt`)
- 位置: `data/raw/GRDC/`
- 获取: https://www.bafg.de/GRDC/

### ISIMIP气候数据
- 格式: NetCDF (`.nc`)
- 位置: `data/raw/ISIMIP3a/`
- 获取: https://www.isimip.org/

## 贡献指南

1. 遵循PEP 8代码规范
2. 所有函数必须包含docstring
3. 新功能需附带单元测试
4. 提交前运行: `black src/` 格式化代码

## 参考文献

1. Budyko, M. I. (1974). *Climate and Life*. Academic Press.
2. Yang et al. (2008). "New analytical derivation of the mean annual water-energy balance equation." *Water Resources Research*, 44, W03410.
3. Xu et al. (2013). "Technical Note: Analytical inversion of the parametric Budyko equations." *Hydrology and Earth System Sciences*, 17, 4397-4404.

## 许可证

MIT License

## 联系方式

- 作者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [GitHub链接]

---

## 开发环境

- **Python**: 3.11.14
- **虚拟环境**: `cat` (conda)
- **环境路径**: `D:\Anaconda3\envs\cat`
- **测试框架**: pytest 9.0.2
- **代码格式化**: black 25.12.0
- **测试覆盖率**: pytest-cov 7.0.0

激活环境:
```bash
conda activate cat
```

---

**最后更新**: 2026-01-01  
**当前版本**: v0.8.0-beta  
**开发环境**: cat (Python 3.11.14)  
**测试状态**: 247/247 通过 ✅  
**GitHub**: https://github.com/SysuCodeRookie/budyko-runoff-attribution
