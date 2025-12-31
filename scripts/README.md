# 真实数据下载与测试指南

本目录包含用于下载真实数据集并在真实数据上测试代码的脚本。

## 📁 脚本说明

### 1. `download_data.py` - 数据下载和准备脚本

**功能**:
- 生成GRDC、ISIMIP、Huang et al. (2018)数据集的下载说明
- 创建模拟测试数据用于快速验证
- 准备数据目录结构

**使用方法**:
```bash
python scripts/download_data.py
```

**输出**:
- `data/raw/GRDC/GRDC_DOWNLOAD_INSTRUCTIONS.txt` - GRDC数据下载详细步骤
- `data/raw/ISIMIP/ISIMIP_DATA_INFO.txt` - ISIMIP数据访问说明
- `data/raw/Huang2018/` - Huang用水数据目录
- `data/raw/GRDC/SAMPLE_STATION_Q_Day.Cmd.txt` - 模拟GRDC数据（1960-2016）

### 2. `test_real_data.py` - 真实数据测试套件

**功能**:
- 测试GRDC数据解析
- 测试完整归因分析工作流
- 测试批量站点处理
- 导出分析结果

**使用方法**:
```bash
python scripts/test_real_data.py
```

**输出**:
- `data/results/test_combined_data.csv` - 合并的径流和气候数据
- `data/results/test_calibration_results.csv` - 参数校准结果
- `data/results/test_attribution_results.csv` - 归因分析结果
- `data/results/test_batch_calibration.csv` - 批量站点校准结果

## 📊 数据集说明

### 1. GRDC观测径流数据

**来源**: Global Runoff Data Center (GRDC)  
**网址**: https://portal.grdc.bafg.de/applications/public.html  
**获取方式**: 需要注册账号

**推荐测试站点（中国流域）**:
- 长江宜昌站 (6335020) - 流域面积较大，数据完整
- 黄河花园口站 (6258300) - 典型半干旱流域
- 珠江石角站 (6381100) - 南方湿润流域

**数据格式**: `*_Q_Day.Cmd.txt` 或 `*_Q_Month.Cmd.txt`

**下载步骤**:
1. 访问GRDC门户并注册账号
2. 搜索站点（可按国家、流域、站点ID搜索）
3. 下载数据文件到 `data/raw/GRDC/`

### 2. ISIMIP3a气候强迫数据

**来源**: Inter-Sectoral Impact Model Intercomparison Project  
**网址**: https://data.isimip.org/  
**获取方式**: 需要注册账号并使用DKRZ数据访问工具

**推荐数据集**: GSWP3-W5E5 (obsclim)

**所需变量**:
- `pr` - 降水量 (kg m⁻² s⁻¹)
- `tas` - 近地面平均气温 (K)
- `tasmax` - 日最高气温 (K)
- `tasmin` - 日最低气温 (K)
- `rsds` - 地表下行短波辐射 (W m⁻²)
- `hurs` - 相对湿度 (%)
- `sfcWind` - 近地面风速 (m s⁻¹)
- `ps` - 表面气压 (Pa)

**时间范围**: 1960-2016

**文件命名示例**:
- `gswp3-w5e5_obsclim_pr_global_daily_1960_2016.nc`
- `gswp3-w5e5_obsclim_tas_global_daily_1960_2016.nc`

**下载步骤**:
1. 访问 https://www.isimip.org/ 并注册账号
2. 登录数据门户 https://data.isimip.org/
3. 选择 ISIMIP3a → GSWP3-W5E5 → obsclim
4. 下载所需变量（建议使用wget或官方下载脚本）
5. 保存到 `data/raw/ISIMIP/`

### 3. Huang et al. (2018)全球用水数据

**来源**: Zenodo开放数据集  
**DOI**: 10.5281/zenodo.1209296  
**网址**: https://zenodo.org/record/1209296  
**获取方式**: 公开可下载（无需注册）

**数据内容**: 1971-2010年全球0.5°分辨率月度分部门用水数据
- 灌溉用水
- 畜牧用水
- 电力用水
- 生活用水
- 矿业用水
- 制造业用水

**数据大小**: >10GB

**下载建议**:
- 可选择下载部分年份或部门数据
- 建议先下载年度总用水量文件进行测试

**下载步骤**:
1. 访问 https://zenodo.org/record/1209296
2. 选择需要的文件（建议从小文件开始）
3. 下载到 `data/raw/Huang2018/`

## 🧪 使用模拟数据快速测试

如果真实数据尚未下载，可以使用自动生成的模拟数据进行测试：

```bash
# 1. 生成模拟数据
python scripts/download_data.py

# 2. 运行测试（使用模拟数据）
python scripts/test_real_data.py
```

模拟数据包括：
- **GRDC模拟数据**: 1960-2016年日径流（20820天）
  - 包含季节周期和年际波动
  - 微弱下降趋势（模拟气候变化影响）
  - 文件: `data/raw/GRDC/SAMPLE_STATION_Q_Day.Cmd.txt`

- **气候模拟数据**: 在`test_real_data.py`中动态生成
  - 降水 P: 850 ± 100 mm/year
  - PET: 1100 ± 120 mm/year
  - 气温: 15 ± 2 °C

## 📈 测试结果解读

运行`test_real_data.py`后会生成以下结果：

### 1. 校准结果 (`test_calibration_results.csv`)

```csv
station_id,n,P,PET,Q_n,E,aridity_index,calibration_error
TEST_STATION,1.47,856.6,1122.6,252.5,604.1,1.31,<0.01%
```

- **n**: 流域景观参数（反映下垫面特征）
- **aridity_index**: 干旱指数（PET/P）
  - < 1.0: 湿润气候（能量限制）
  - > 1.0: 干旱气候（水分限制）
- **calibration_error**: 校准误差（应<1%）

### 2. 归因结果 (`test_attribution_results.csv`)

```csv
station_id,delta_Q_obs,C_CCV,C_LUCC,C_WADR
TEST_STATION,-23.15,54.7%,-5.4%,51.0%
```

- **C_CCV**: 气候变化贡献率
- **C_LUCC**: 土地利用变化贡献率
- **C_WADR**: 人类取用水贡献率

**解读示例**:
- 径流减少23.15 mm/year
- 气候变化贡献54.7%（主导因子）
- 人类取用水贡献51.0%
- 土地利用变化贡献-5.4%（蒸散发略有减少）

### 3. 批量校准结果 (`test_batch_calibration.csv`)

展示不同气候类型流域的参数特征：
- **湿润流域**: n较小（~1.3），aridity_index < 1
- **干旱流域**: n更小（~0.9），aridity_index > 4

## 🔧 故障排除

### 问题1: GRDC数据解析失败

**原因**: 文件格式不正确或不是真实GRDC文件

**解决方案**:
- 确保文件来自GRDC官方
- 检查文件头是否包含元数据（以`# GRDC-No.:`开头）
- 使用`SAMPLE_STATION_Q_Day.Cmd.txt`测试解析器功能

### 问题2: 气候数据缺失

**原因**: ISIMIP数据尚未下载

**解决方案**:
- 使用模拟数据进行测试（自动生成）
- 按照说明下载ISIMIP数据
- 或使用项目的`ClimateDataProcessor`模块处理其他格式气候数据

### 问题3: 测试结果不合理

**原因**: 模拟数据的随机性

**解决方案**:
- 模拟数据仅用于功能测试，不代表真实物理过程
- 使用真实数据获得科学有效的归因结果
- 检查水量平衡（Q_n < P应始终成立）

## 📝 下一步工作

使用真实数据进行完整分析：

1. **准备数据**:
   ```bash
   # 下载真实GRDC、ISIMIP数据后
   python scripts/test_real_data.py
   ```

2. **批量站点分析**:
   - 准备多个站点的GRDC数据
   - 提取对应的ISIMIP气候数据
   - 运行批量归因分析

3. **区域尺度研究**:
   - 使用`calculate_ensemble_attribution()`综合多站点结果
   - 分析区域平均归因贡献
   - 生成可视化图表

## 📚 参考文献

1. **GRDC**: BfG (2024). The Global Runoff Data Centre. Federal Institute of Hydrology (BfG), Koblenz, Germany.

2. **ISIMIP**: Lange, S., et al. (2020). ISIMIP3a atmospheric climate input data (v1.0). ISIMIP Repository.

3. **Huang et al. (2018)**: Huang, Z., Hejazi, M., Li, X., et al. (2018). Reconstruction of global gridded monthly sectoral water withdrawals for 1971–2010 and analysis of their spatiotemporal patterns. Hydrology and Earth System Sciences, 22(4), 2117-2133.

---

**最后更新**: 2025-01-01  
**维护者**: Research Software Engineer  
**项目**: Budyko径流归因分析系统
