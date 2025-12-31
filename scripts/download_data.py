"""
download_data.py

真实数据下载脚本

下载并准备以下数据集：
1. GRDC观测径流数据（需要手动注册下载）
2. ISIMIP3a气候强迫数据（GSWP3-W5E5）
3. Huang et al. (2018)全球用水数据

作者: Research Software Engineer
日期: 2025-01-01
"""

import os
import sys
from pathlib import Path
import zipfile
import gzip
import shutil
from urllib.parse import urljoin

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  Warning: requests库未安装，无法自动下载文件")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DataDownloader:
    """真实数据集下载器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化下载器
        
        Args:
            data_dir: 数据存储根目录
        """
        if data_dir is None:
            data_dir = project_root / "data" / "raw"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.grdc_dir = self.data_dir / "GRDC"
        self.isimip_dir = self.data_dir / "ISIMIP"
        self.huang_dir = self.data_dir / "Huang2018"
        
        for d in [self.grdc_dir, self.isimip_dir, self.huang_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def download_file(
        self, 
        url: str, 
        output_path: Path,
        description: str = "Downloading"
    ) -> bool:
        """
        下载文件（带进度条）
        
        Args:
            url: 下载URL
            output_path: 输出文件路径
            description: 下载描述
        
        Returns:
            是否下载成功
        """
        if not REQUESTS_AVAILABLE:
            print(f"❌ 无法下载: requests库未安装")
            print(f"   请手动下载: {url}")
            return False
        
        try:
            print(f"\n{description}...")
            print(f"URL: {url}")
            print(f"保存到: {output_path}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if not TQDM_AVAILABLE and total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r进度: {percent:.1f}%", end='', flush=True)
            
            if not TQDM_AVAILABLE:
                print()  # 换行
            
            print(f"✅ 下载成功: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ 下载失败: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_huang_2018_data(self):
        """
        下载Huang et al. (2018)全球用水数据
        
        数据来源: https://zenodo.org/record/1209296
        DOI: 10.5281/zenodo.1209296
        
        包含1971-2010年全球0.5度分辨率的月度分部门用水数据
        """
        print("\n" + "="*70)
        print("下载 Huang et al. (2018) 全球用水数据")
        print("="*70)
        
        # Zenodo直接下载链接
        zenodo_base = "https://zenodo.org/record/1209296/files/"
        
        # 数据文件列表（示例：下载主要文件）
        files_to_download = [
            ("wateruse_1971_2010_annual.nc", "年度总用水量"),
            # 根据实际需要添加更多文件
        ]
        
        print("\n⚠️  注意：Huang et al. (2018)数据集较大（>10GB）")
        print("建议先下载样本数据或特定年份数据")
        print("\n完整数据集访问：https://zenodo.org/record/1209296")
        
        # 尝试下载README
        readme_url = zenodo_base + "README.txt"
        readme_path = self.huang_dir / "README.txt"
        
        if self.download_file(readme_url, readme_path, "下载README"):
            print("\n数据集信息:")
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                print(f.read()[:500])
        
        print("\n建议手动下载数据文件：")
        print("1. 访问: https://zenodo.org/record/1209296")
        print("2. 选择需要的文件下载到:", self.huang_dir)
        
        return True
    
    def download_isimip_sample_data(self):
        """
        下载ISIMIP3a示例数据
        
        ISIMIP数据量巨大（>TB级），这里下载小样本用于测试
        完整数据需要通过ISIMIP官方门户申请
        """
        print("\n" + "="*70)
        print("ISIMIP3a 气候数据")
        print("="*70)
        
        print("\n⚠️  ISIMIP3a完整数据集需要通过官方门户访问：")
        print("   https://data.isimip.org/")
        print("   需要注册账号并使用DKRZ数据访问工具")
        
        # 可以尝试下载一些公开的小样本或文档
        isimip_info = """
ISIMIP3a 数据获取步骤：

1. 注册账号：https://www.isimip.org/account/register/
2. 访问数据门户：https://data.isimip.org/
3. 选择数据：
   - Simulation Round: ISIMIP3a
   - Climate Forcing: GSWP3-W5E5 (obsclim)
   - 变量: pr (降水), tas (气温), rsds (辐射), hurs (湿度), sfcWind (风速)
   - 时间范围: 1960-2016
4. 下载工具：使用wget或DKRZ提供的下载脚本

示例文件名：
- gswp3-w5e5_obsclim_pr_global_daily_1960_2016.nc
- gswp3-w5e5_obsclim_tas_global_daily_1960_2016.nc
"""
        
        info_path = self.isimip_dir / "ISIMIP_DATA_INFO.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(isimip_info)
        
        print(isimip_info)
        print(f"\n信息已保存到: {info_path}")
        
        return True
    
    def setup_grdc_instructions(self):
        """
        生成GRDC数据下载说明
        
        GRDC数据需要注册账号才能下载
        """
        print("\n" + "="*70)
        print("GRDC 观测径流数据")
        print("="*70)
        
        grdc_info = """
GRDC（全球径流数据中心）数据获取步骤：

1. 注册账号：
   访问: https://portal.grdc.bafg.de/applications/public.html
   点击 "Register" 创建账号

2. 登录后搜索站点：
   - 可按国家、流域、站点ID搜索
   - 建议选择长序列（>50年）、数据完整度高的站点

3. 下载数据：
   - 选择站点后点击下载
   - 文件格式: *_Q_Day.Cmd.txt 或 *_Q_Month.Cmd.txt
   - 保存到: {grdc_dir}

4. 推荐测试站点（中国流域）：
   - 长江宜昌站 (6335020)
   - 黄河花园口站 (6258300)
   - 珠江石角站 (6381100)

5. 元数据说明：
   - GRDC文件头包含站点元数据（坐标、面积等）
   - 本项目的GRDCParser可自动解析

示例文件：
- 6335020_Q_Day.Cmd.txt  (长江宜昌站日径流)
- 6258300_Q_Month.Cmd.txt (黄河花园口站月径流)
"""
        
        info_path = self.grdc_dir / "GRDC_DOWNLOAD_INSTRUCTIONS.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(grdc_info.format(grdc_dir=self.grdc_dir))
        
        print(grdc_info.format(grdc_dir=self.grdc_dir))
        print(f"\n说明已保存到: {info_path}")
        
        return True
    
    def create_sample_data_for_testing(self):
        """
        创建模拟数据用于快速测试
        
        当真实数据尚未下载时，可以使用模拟数据测试代码功能
        """
        print("\n" + "="*70)
        print("生成模拟测试数据")
        print("="*70)
        
        import numpy as np
        import pandas as pd
        
        # 创建模拟GRDC数据
        print("\n生成模拟GRDC数据...")
        grdc_sample = self.grdc_dir / "SAMPLE_STATION_Q_Day.Cmd.txt"
        
        grdc_content = """# GRDC-No.:                        9999999
# River:                          Sample River
# Station:                        Test Station
# Country:                        CN
# Latitude (DD):                  30.50
# Longitude (DD):                 110.25
# Catchment area (km�):           100000.0
# Altitude (m ASL):               50
# Next downstream station:        -
# Remarks:                        Sample data for testing
# Owner of original data:         Test Organization
#************************************************************
# Data Set Content:               MEAN DAILY DISCHARGE (Q)
# Data Set Content:               ��
# Unit of measure:                m�/s
# Time series:                    1960-01 - 2016-12
# No. of years:                   57
# Last update:                    2025-01-01
# Calculated from:                daily data
# Publication level:              Free
#************************************************************
# YYYY-MM-DD    hh:mm     Value    OC
# Instantaneous discharge (m�/s)
#************************************************************
"""
        
        # 生成1960-2016年的模拟日径流数据
        start_date = pd.date_range('1960-01-01', '2016-12-31', freq='D')
        np.random.seed(123)
        
        # 模拟年内季节变化和年际波动
        days = np.arange(len(start_date))
        seasonal = 5000 + 3000 * np.sin(2 * np.pi * days / 365.25)  # 季节周期
        trend = -0.5 * days / 365.25  # 微弱下降趋势
        noise = np.random.normal(0, 500, len(start_date))
        discharge = seasonal + trend + noise
        discharge = np.maximum(discharge, 100)  # 确保非负
        
        with open(grdc_sample, 'w', encoding='utf-8') as f:
            f.write(grdc_content)
            for date, q in zip(start_date, discharge):
                f.write(f"{date.strftime('%Y-%m-%d')}; 00:00; {q:.1f}; \n")
        
        print(f"✅ 创建: {grdc_sample}")
        print(f"   包含 {len(start_date)} 天数据 (1960-2016)")
        
        # 创建模拟气候数据说明
        print("\n生成模拟气候数据说明...")
        climate_note = self.isimip_dir / "SAMPLE_DATA_NOTE.txt"
        with open(climate_note, 'w', encoding='utf-8') as f:
            f.write("模拟气候数据可通过 ClimateProcessor 的示例方法生成\n")
            f.write("参见: examples/climate_processor_example.py\n")
        
        print(f"✅ 创建: {climate_note}")
        
        print("\n✅ 模拟数据生成完成！")
        print("可以使用这些数据测试代码功能")
        
        return True
    
    def run_all(self):
        """执行所有数据准备步骤"""
        print("\n" + "="*70)
        print("真实数据下载与准备工具")
        print("="*70)
        print(f"\n数据根目录: {self.data_dir}")
        
        # 1. GRDC说明
        self.setup_grdc_instructions()
        
        # 2. ISIMIP信息
        self.download_isimip_sample_data()
        
        # 3. Huang数据
        self.download_huang_2018_data()
        
        # 4. 创建模拟测试数据
        self.create_sample_data_for_testing()
        
        print("\n" + "="*70)
        print("数据准备总结")
        print("="*70)
        print(f"""
✅ 已完成:
   - GRDC下载说明生成
   - ISIMIP访问信息提供
   - Huang数据集信息提供
   - 模拟测试数据创建

📁 数据目录结构:
   {self.grdc_dir}/     - GRDC观测径流（需手动下载）
   {self.isimip_dir}/   - ISIMIP气候数据（需注册下载）
   {self.huang_dir}/    - Huang用水数据（可从Zenodo下载）

🔍 下一步:
   1. 按照说明文件手动下载GRDC和ISIMIP数据
   2. 使用模拟数据测试代码: python scripts/test_real_data.py
   3. 替换为真实数据后进行完整分析
""")


def main():
    """主函数"""
    downloader = DataDownloader()
    downloader.run_all()


if __name__ == "__main__":
    main()
