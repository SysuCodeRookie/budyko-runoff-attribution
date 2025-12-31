# **基于Budyko假设的径流归因分析方法Python实现技术指南与详细说明文档**

## **1\. 执行摘要与项目背景**

本技术报告旨在为开发一套基于Python的径流归因分析计算框架提供详尽的理论依据、算法逻辑及代码实现指南。作为土木工程及水文学领域的博士研究工作的一部分，本项目核心任务是将经典的Budyko水热耦合平衡理论与现代多源水文气象数据相结合，量化气候变化（Climate Change and Variability, CCV）、土地利用/覆盖变化（Land Use/Cover Change, LUCC）以及人类取用水及水库调蓄（Water Abstraction, Diversion, and Regulation, WADR）对河川径流变化的相对贡献。

本报告并非直接提供源代码，而是作为一份高级技术需求规格说明书，指导后续代码编写人员（Research Software Engineers）理解背后的物理机制、数学推导及数据处理细节。我们将依据上传的main.tex文档中描述的方法论，结合ISIMIP3a（Inter-Sectoral Impact Model Intercomparison Project Phase 3a）协议、GRDC（Global Runoff Data Center）观测数据以及Huang et al. (2018)构建的全球用水数据集，构建一个模块化、可扩展的归因分析流水线。

报告将首先深入剖析Budyko假设及其解析形式（Choudhury-Yang方程）的物理内涵，详述径流弹性的解析推导过程；其次，系统梳理多源异构数据的特征、预处理要求及空间匹配策略；随后，将核心计算任务分解为六大关键步骤，并针对每一步骤提供具体的算法伪代码逻辑与Python库选型建议；最后，探讨模型验证、不确定性分析及软件工程最佳实践，以确保计算结果的科学性与复现性。

## ---

**2\. 理论框架与数学物理基础**

在编写任何代码之前，开发团队必须深刻理解支配代码逻辑的水文物理方程。本项目的核心在于利用Budyko框架解释流域尺度的长期水能量平衡。

### **2.1 Budyko假设与Choudhury-Yang方程**

Budyko假设认为，在长时间尺度（通常为多年平均）下，流域的实际蒸散发（$E$）主要受限于大气的供水能力（降水，$P$）和供能能力（潜在蒸散发，$PET$）。这一假设是水文学中“顶层设计”方法的基石，它不仅物理意义明确，而且对数据需求相对较低，非常适合大尺度流域的归因分析 1。

虽然原始的Budyko曲线是无参数的，但为了捕捉不同流域下垫面特征（如植被覆盖、土壤性质、地形坡度等）对水热平衡的特异性影响，本项目采用**Choudhury-Yang方程**。这是一种参数化的Budyko公式，通过引入特定的下垫面参数$n$来修正曲线形状 1。

代码实现的核心公式为：

$$E \= \\frac{P \\times PET}{(P^n \+ PET^n)^{\\frac{1}{n}}}$$  
其中：

* **$E$ (mm/year)**: 多年平均实际蒸散发。  
* **$P$ (mm/year)**: 多年平均降水量。  
* **$PET$ (mm/year)**: 多年平均潜在蒸散发，表征大气的蒸发需求。  
* **$n$ (无量纲)**: 流域景观特征参数（Landscape Parameter），集成反映了土壤渗透性、植被截留与蒸腾能力、地形坡度等因素。当$n$增大时，意味着流域截留雨水转化为蒸散发的能力增强；反之，则更倾向于产流。

基于流域水量平衡原理，假设在多年尺度上流域陆地水储量变化（$\\Delta S$）趋近于零（$\\Delta S \\approx 0$），则天然径流（$Q\_n$）可表示为降水与蒸散发之差 1：

$$Q\_n \= P \- E \= P \- \\frac{P \\times PET}{(P^n \+ PET^n)^{\\frac{1}{n}}}$$  
**代码编写关键点：**

* 该方程不仅用于由已知参数计算径流（正向过程），更关键的是用于**反向演算**。即已知多年平均的观测数据$P$、$PET$和$Q\_n$，利用数值优化算法求解参数$n$。这是模型校准的核心步骤。  
* 公式中的幂运算涉及非整数指数，需注意底数非负性检查，防止程序抛出复数域错误或运行时异常。

### **2.2 径流弹性系数的解析推导**

本项目采用弹性系数法（Elasticity Method）进行归因，该方法基于偏微分原理，量化径流对各驱动因子微小变化的敏感性。假设$P$、$PET$和$n$是相互独立的变量，根据全微分公式，径流的变化率可分解为 1：

$$\\frac{dQ\_n}{Q\_n} \= \\varepsilon\_P \\frac{dP}{P} \+ \\varepsilon\_{PET} \\frac{dPET}{PET} \+ \\varepsilon\_n \\frac{dn}{n}$$  
其中，$\\varepsilon\_x$表示径流对变量$x$的弹性系数，定义为$\\frac{\\partial Q\_n / Q\_n}{\\partial x / x}$。代码必须实现以下三个解析解的计算逻辑：

1\. 降水弹性系数 ($\\varepsilon\_P$)：  
反映降水变化1%导致的径流变化百分比。通常$\\varepsilon\_P \> 1$，表明径流对降水具有放大效应。

$$\\varepsilon\_P \= \\frac{1 \- \\left^{\\frac{1}{n} \+ 1}}{1 \- \\left^{\\frac{1}{n}}}$$  
2\. 潜在蒸散发弹性系数 ($\\varepsilon\_{PET}$)  
反映气温升高、辐射增强等导致的大气需水增加对径流的削减作用。通常$\\varepsilon\_{PET} \< 0$。

$$\\varepsilon\_{PET} \= \\frac{1}{1 \+ \\left(\\frac{PET}{P}\\right)^n} \\frac{1}{1 \- \\left^{\\frac{1}{n}}}$$  
3\. 下垫面参数弹性系数 ($\\varepsilon\_n$)  
反映植被变化或人类活动导致的景观改变对径流的影响。通常$\\varepsilon\_n \< 0$，即下垫面截留能力增强（如植林）会导致径流减少。

$$\\varepsilon\_n \= \\frac{1}{\\left^{\\frac{1}{n}} \- 1} \\left$$  
**算法实现建议：**

* 为避免代码中重复计算复杂项，建议引入中间变量（如干旱指数 $\\phi \= PET/P$）。  
* 注意 $\\varepsilon\_P \+ \\varepsilon\_{PET}$ 的关系。在某些Budyko公式形式下，两者之和为1。但在包含参数$n$的Choudhury-Yang公式中，这一关系可能更为复杂，需作为一种软约束进行结果合理性检验。  
* 公式中包含自然对数 $\\ln$，需确保输入值大于0。对于极度干旱地区或极端数据，需加入极小值平滑处理（epsilon smoothing）。

### **2.3 归因分解逻辑**

归因分析的核心是将观测到的径流变化（$\\Delta Q\_o$）拆解为不同驱动因子的贡献。根据main.tex文档，分解路径如下 1：

1. 气候变化与变率 (CCV) 的贡献：  
   通过降水和PET的变化量（$\\Delta P, \\Delta PET$）乘以相应的弹性系数计算。

   $$\\Delta Q\_{n,CCV} \= \\varepsilon\_P \\frac{Q\_n}{P} \\Delta P \+ \\varepsilon\_{PET} \\frac{Q\_n}{PET} \\Delta PET$$  
2. 土地利用/覆盖变化 (LUCC) 的贡献：  
   通过下垫面参数的变化量（$\\Delta n$）乘以其弹性系数计算。

   $$\\Delta Q\_{n,LUCC} \= \\varepsilon\_n \\frac{Q\_n}{n} \\Delta n$$

   注：这里的$\\Delta n$反映了除气候要素外的流域属性变化，主要归因于LUCC。  
3. 人类取用水及调节 (WADR) 的贡献：  
   这一项通常难以通过弹性方法直接计算，而是作为“残差项”或通过观测径流与天然径流的差值来确定。

   $$C\_{WADR} \= \\frac{\\Delta Q\_o \- \\Delta Q\_n}{\\Delta Q\_o} \\times 100\\%$$

   这里隐含了一个关键的数据处理步骤：必须通过还原计算获得天然径流$Q\_n$，与实测径流$Q\_o$对比，从而剥离出WADR的直接影响。

## ---

**3\. 数据源详述与工程化处理策略**

代码的健壮性在很大程度上取决于数据处理流水线（Pipeline）的设计。本项目涉及三种异构数据源，它们在空间分辨率、时间步长和文件格式上各不相同，需要进行严格的清洗、对齐和转换。

### **3.1 观测径流数据 (Observed Streamflow, $Q\_o$)**

* **数据来源**：全球径流数据中心 (GRDC, Global Runoff Data Center) 1。  
* **数据特征**：通常为文本格式（.mon）或专有格式，记录了各水文站点的日或月平均流量（$m^3/s$）。  
* **处理任务**：  
  1. **元数据解析**：从文件头提取站点ID、经纬度坐标、集水区面积（Catchment Area）。  
  2. 单位转换：Budyko方程基于水深（mm），而GRDC数据为流量（$m^3/s$）。代码必须实现基于集水区面积的转换函数：

     $$R\_{depth} (mm) \= \\frac{Q\_{vol} (m^3/s) \\times \\Delta t (s)}{Area (km^2) \\times 1000}$$  
  3. **缺测处理**：对于存在缺失值的年份，需设定严格的筛选标准（例如，若某年缺失数据超过15%，则该年标记为无效）。  
  4. **时间聚合**：将日/月数据聚合为**水文年**或日历年的年平均值/年总值。

### **3.2 气象驱动数据 ($P, PET$)**

* **数据来源**：ISIMIP3a (Inter-Sectoral Impact Model Intercomparison Project) 的 obsclim 数据集，具体为 **GSWP3-W5E5** 大气强迫数据 1。  
* **数据特征**：全球网格化数据（通常为0.5° $\\times$ 0.5°分辨率），NetCDF格式（.nc）。  
* **关键变量**：  
  * pr: 降水通量 ($kg \\cdot m^{-2} \\cdot s^{-1}$)，需转换为年降水量 ($mm/year$)。  
  * tas, tasmax, tasmin: 近地面气温 (K)，需转换为摄氏度 ($^\\circ C$) 用于PET计算。  
  * rsds: 地表下行短波辐射 ($W \\cdot m^{-2}$)。  
  * hurs: 相对湿度 (%)。  
  * sfcWind: 近地面风速 ($m \\cdot s^{-1}$)。  
  * ps: 表面气压 (Pa)。  
* **PET计算模块**：ISIMIP3a原始数据通常不直接提供FAO-56定义的PET。代码需集成pyet或pyfao56库，利用上述气象变量计算日尺度的参考作物蒸散发（Reference Evapotranspiration, $ET\_0$），并将其作为$PET$输入到Budyko方程中 10。  
* **空间掩膜（Masking）**：这是最具挑战性的步骤。必须根据GRDC站点的集水区边界（通常为Shapefile多边形），提取对应的ISIMIP网格数据，并计算**面积加权平均值**（Area-Weighted Mean）。这需要使用geopandas和rioxarray进行精确的空间叠置分析 7。

### **3.3 人类用水数据 (Water Consumption)**

* **数据来源**：Huang et al. (2018) 重建的全球网格化月度分部门用水数据集 1。  
* **数据特征**：NetCDF格式，包含不同部门（灌溉、家畜、电力、生活、矿业、制造）的取水量（Withdrawal）和耗水量（Consumption）。  
* **处理逻辑**：  
  1. **天然径流还原**：$Q\_n$ 的计算公式通常定义为：$Q\_n \= Q\_o \+ W\_{consumption} \+ \\Delta S\_{reservoir}$。但在本研究中，根据文档描述，主要是加上耗水量（$W\_{consumption}$）来还原。代码需读取各部门的耗水数据，求和得到总耗水。  
  2. **网格映射**：Huang et al. 数据的分辨率可能与ISIMIP不一致，代码需具备重采样（Resampling）或对齐（Regridding）功能，确保其能正确聚合到GRDC流域边界内。  
  3. **WADR量化**：该数据直接支持WADR（取水、引水、调蓄）贡献的计算。通过比较$Q\_n$和$Q\_o$的差异，量化直接人为干预的影响。

### **3.4 ISIMIP3a 模型模拟数据 (可选/验证用)**

* **数据来源**：ISIMIP3a 全球水文模型（GHMs）输出 1。  
* **实验情景**：  
  * obsclim \+ histsoc ($Q'\_o$): 模拟的观测径流。  
  * obsclim \+ 1901soc ($Q'\_n$): 模拟的天然化径流（仅含气候变化，固定1901年人类活动）。  
  * counterclim \+ 1901soc ($Q'\_{cn}$): 模拟的基准径流（去趋势气候 \+ 固定人类活动）。  
* **用途**：用于进一步分离气候变化中的**人为气候变化（ACC）和自然气候变率（NCV）**。这是一个高级功能，代码应设计为可插拔模块。

## ---

**4\. 代码撰写详细建议与任务分解**

为了保证代码的结构清晰、易于维护和扩展，建议采用面向对象编程（OOP）范式，将不同的物理过程和数据处理步骤封装为独立的类。以下是详细的任务分解：

### **任务一：构建数据预处理引擎 (Data Preprocessing Engine)**

**目标**：将多源异构原始数据转换为统一格式的结构化数据（如Pandas DataFrame或Xarray Dataset）。

* **步骤 1.1：开发 GRDCParser 类**  
  * **功能**：读取GRDC文本文件，解析Header信息（面积、坐标），清洗时间序列。  
  * **关键点**：实现异常值检测（如负值流量处理），并自动完成 $m^3/s \\to mm/year$ 的单位转换。  
* **步骤 1.2：开发 ClimateDataProcessor 类**  
  * **功能**：处理ISIMIP NetCDF文件。  
  * **关键点**：实现“空间聚合算法”。输入为流域Shapefile和气候NetCDF，输出为该流域的年平均气候序列。建议使用 rioxarray.clip 或 rasterstats 库来处理不规则边界的多边形掩膜提取。  
* **步骤 1.3：集成 PETCalculator 模块**  
  * **功能**：基于日尺度气象数据计算FAO-56 Penman-Monteith PET。  
  * **建议**：不要从零手写物理公式，直接调用 pyet 库，确保参数（如反照率、冠层阻力）设置符合FAO标准（参考作物：短草）。  
* **步骤 1.4：开发 WaterUseAggregator 类**  
  * **功能**：处理Huang et al. (2018) 数据，聚合各部门耗水，并匹配到相应流域。  
  * **关键点**：计算天然径流序列 $Q\_n \= Q\_o \+ W\_{consumption}$。

### **任务二：核心归因算法实现 (Core Attribution Solver)**

**目标**：实现Budyko方程的参数率定和弹性系数计算。

* **步骤 2.1：实现 BudykoModel 类**  
  * **方法 calibrate\_n(P, PET, Q\_n)**：  
    * 这是一个反演问题。定义目标函数 $f(n) \= Q\_{n, simulated}(P, PET, n) \- Q\_{n, observed}$。  
    * 使用 scipy.optimize.brentq 或 newton 方法寻找使 $f(n)=0$ 的 $n$ 值。  
    * **约束条件**：$n$ 必须为正数（通常在0.1到10之间）。需添加 try-except 块处理无解或发散的情况（例如当 $Q\_n \> P$ 时，违反水量平衡，需报错或标记）。  
  * **方法 calculate\_elasticities(P, PET, n)**：  
    * 直接翻译2.2节中的三个解析公式。  
    * **验证**：在单元测试中检查 $\\varepsilon\_P$ 是否主要为正，$\\varepsilon\_{PET}$ 是否为负。

### **任务三：归因分析工作流 (Attribution Workflow)**

**目标**：串联上述模块，执行时段对比分析。

* **步骤 3.1：时段划分 (Period Segmentation)**  
  * 根据文档，需支持灵活的突变点设置（如1986年）。代码应将时间序列划分为 Pre-Change (Base) 和 Post-Change (Impact) 两个阶段。  
* **步骤 3.2：执行归因计算**  
  * 计算两个时段的均值：$\\overline{P}\_{pre}, \\overline{P}\_{post}, \\overline{PET}\_{pre}, \\dots$  
  * 计算变化量：$\\Delta P, \\Delta PET, \\Delta n$。注意 $\\Delta n$ 的计算需要分别在两个时段反演 $n$ 值。  
  * 调用弹性系数计算模块（通常使用全序列或基准期的均值计算弹性）。  
  * 计算 $\\Delta Q\_{CCV}$ 和 $\\Delta Q\_{LUCC}$。  
  * 计算 $C\_{WADR}$。  
* **步骤 3.3：结果汇总与输出**  
  * 生成包含每个站点归因结果的CSV文件（站点ID, $\\Delta Q$, $C\_{CCV}\\%$, $C\_{LUCC}\\%$, $C\_{WADR}\\%$ 等）。

### **任务四：高级归因与ISIMIP集成 (Advanced ISIMIP Integration)**

**目标**：利用ISIMIP模型输出分离气候变化信号。

* **步骤 4.1：解析ISIMIP文件名与场景**  
  * 编写逻辑识别 obsclim, counterclim, histsoc, 1901soc 文件。  
* **步骤 4.2：计算 ACC 与 NCV**  
  * $\\Delta Q'\_{n}$ (由 obsclim \+ 1901soc 驱动) 代表总气候影响 (ACC \+ NCV)。  
  * $\\Delta Q'\_{cn}$ (由 counterclim \+ 1901soc 驱动) 代表自然变率 (NCV)。  
  * 通过差值计算人为气候变化 (ACC) 的贡献。

## ---

**5\. 关键技术难点与应对策略 (Technical Nuances)**

在代码实现过程中，极易遇到以下“隐形陷阱”，必须在开发文档中予以强调：

1. 水量平衡违背问题：  
   在某些数据质量较差或存在跨流域调水的站点，可能出现 $Q\_n \> P$ 的情况。这在物理上意味着蒸散发为负，导致Budyko方程无解。  
   * **策略**：代码必须包含“物理一致性检查（Sanity Check）”模块。对于 $Q\_n/P \\ge 1$ 的年份或站点，应自动剔除并记录日志，不可强行计算。  
2. 时间尺度的匹配：  
   Budyko假设严格来说仅适用于多年平均（稳态）。但在归因分析中，我们通常对“年际变化”进行操作。  
   * **策略**：虽然计算是逐年进行的，但在参数率定（计算$n$）时，必须使用长序列的平均值。切勿对每一年的数据单独反算$n$，这会引入巨大的噪声。  
3. 空间分辨率差异：  
   Huang et al. (2018) 的用水数据分辨率可能与ISIMIP (0.5度) 不完全一致，且流域边界是不规则多边形。  
   * **策略**：推荐使用 xarray 的 reindex 或 interp 功能将所有网格数据统一重采样到同一分辨率（建议0.5度），然后再进行掩膜提取。或者使用 ExactExtract 等高精度工具计算多边形内的加权平均，以避免边界网格（由陆地和海洋混合）导致的误差。  
4. 分母为零的风险：  
   在计算贡献率百分比时，分母是 $\\Delta Q\_o$。如果某站点在两个时段间径流变化极小（$\\Delta Q\_o \\approx 0$），会导致贡献率趋于无穷大。  
   * **策略**：设置阈值（如 $|\\Delta Q\_o| \< 1mm$），对于变化不显著的站点，不计算百分比贡献，仅输出绝对量（mm），并在报告中标记。

## **6\. 推荐使用的Python库栈**

为了构建专业级的科研代码，建议使用以下技术栈：

| 库名称 | 用途 | 理由 |
| :---- | :---- | :---- |
| **xarray** | 多维数据处理 | 处理NetCDF气象数据的标准库，支持标签索引和广播运算。 |
| **rioxarray / geopandas** | 空间分析 | 能够处理坐标系投影、Shapefile读取及掩膜提取。 |
| **scipy** | 数值计算 | scipy.optimize提供了稳定的一维求根算法（Brent方法）。 |
| **pyet** | 蒸散发计算 | 经过验证的开源库，避免手动编写Penman-Monteith公式出错。 |
| **dask** | 并行计算 | 如果处理全球数千个站点，Dask可实现内存外计算和多核并行。 |
| **matplotlib / seaborn** | 可视化 | 用于绘制归因结果的堆叠柱状图（如Wang 2025中的图表）。 |

## ---

**7\. 结语**

本说明文档提供了一个从理论到实践的完整蓝图。代码撰写人员应严格遵循上述数学推导和数据处理规范，特别注意物理约束条件的检查和异常数据的处理。最终交付的代码不仅是一个计算工具，更是一个能够经得起同行评审（Peer Review）的科学仪器，能够准确地将复杂的流域水文变化解构为清晰的气候与人类活动信号。这对于理解全球变化背景下的水资源演变规律具有重要的学术价值。

#### **引用的著作**

1. main.tex  
2. Understanding interactions among climate, water, and vegetation with the Budyko framework \- Southern Research Station, 访问时间为 十二月 31, 2025， [https://www.srs.fs.usda.gov/pubs/ja/2021/ja\_2021\_sun\_004.pdf](https://www.srs.fs.usda.gov/pubs/ja/2021/ja_2021_sun_004.pdf)  
3. Revisiting the Hydrological Basis of the Budyko Framework With the Hydrologically Similar Groups Principle \- ResearchGate, 访问时间为 十二月 31, 2025， [https://www.researchgate.net/publication/363233463\_Revisiting\_the\_Hydrological\_Basis\_of\_the\_Budyko\_Framework\_With\_the\_Hydrologically\_Similar\_Groups\_Principle/fulltext/63125b40acd814437ff9e06b/Revisiting-the-Hydrological-Basis-of-the-Budyko-Framework-With-the-Hydrologically-Similar-Groups-Principle.pdf](https://www.researchgate.net/publication/363233463_Revisiting_the_Hydrological_Basis_of_the_Budyko_Framework_With_the_Hydrologically_Similar_Groups_Principle/fulltext/63125b40acd814437ff9e06b/Revisiting-the-Hydrological-Basis-of-the-Budyko-Framework-With-the-Hydrologically-Similar-Groups-Principle.pdf)  
4. Dynamic evolution of attribution analysis of runoff based on the complementary Budyko equation in the source area of Lancang river \- Frontiers, 访问时间为 十二月 31, 2025， [https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1160520/full](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1160520/full)  
5. Technical Note: Analytical Inversion of the Parametric Budyko Equations \- ResearchGate, 访问时间为 十二月 31, 2025， [https://www.researchgate.net/publication/347027384\_Technical\_Note\_Analytical\_Inversion\_of\_the\_Parametric\_Budyko\_Equations](https://www.researchgate.net/publication/347027384_Technical_Note_Analytical_Inversion_of_the_Parametric_Budyko_Equations)  
6. Measuring Hydrology | Minnesota DNR, 访问时间为 十二月 31, 2025， [https://www.dnr.state.mn.us/whaf/about/5-component/hydro\_measure.html](https://www.dnr.state.mn.us/whaf/about/5-component/hydro_measure.html)  
7. GRDC-Caravan: extending Caravan with data from the Global Runoff Data Centre, 访问时间为 十二月 31, 2025， [https://essd.copernicus.org/articles/17/4613/2025/](https://essd.copernicus.org/articles/17/4613/2025/)  
8. ISIMIP3a protocol for, 访问时间为 十二月 31, 2025， [https://protocol.isimip.org/](https://protocol.isimip.org/)  
9. Enhancing evapotranspiration estimates under climate change: the role of CO2 physiological feedback and CMIP6 scenarios \- HESS, 访问时间为 十二月 31, 2025， [https://hess.copernicus.org/articles/29/5645/2025/hess-29-5645-2025.pdf](https://hess.copernicus.org/articles/29/5645/2025/hess-29-5645-2025.pdf)  
10. Potential Evapotranspiration (PET) \- Weather \- Texas A\&M University, 访问时间为 十二月 31, 2025， [https://etweather.tamu.edu/pet/](https://etweather.tamu.edu/pet/)  
11. pyet \- Estimation of Potential Evapotranspiration — pyet 2020 documentation, 访问时间为 十二月 31, 2025， [https://pyet.readthedocs.io/](https://pyet.readthedocs.io/)  
12. Supporting Information for: \[1em\] Exploring the Potential of LPJmL-5 to Simulate Vegetation Responses to (Multi-Year) Droughts \- EGUsphere, 访问时间为 十二月 31, 2025， [https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4966/egusphere-2025-4966-supplement.pdf](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4966/egusphere-2025-4966-supplement.pdf)  
13. Caucasus\_Erosion/budyko.py at main \- GitHub, 访问时间为 十二月 31, 2025， [https://github.com/amforte/Caucasus\_Erosion/blob/main/budyko.py](https://github.com/amforte/Caucasus_Erosion/blob/main/budyko.py)  
14. Global gridded monthly sectoral water use dataset for 1971-2010 v2 | Aquaknow, 访问时间为 十二月 31, 2025， [https://aquaknow.jrc.ec.europa.eu/node/18900](https://aquaknow.jrc.ec.europa.eu/node/18900)  
15. Global gridded monthly sectoral water use dataset for 1971-2010: v2 \- Zenodo, 访问时间为 十二月 31, 2025， [https://zenodo.org/record/1209296](https://zenodo.org/record/1209296)