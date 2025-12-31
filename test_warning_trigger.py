import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from tests.test_isimip_attribution import generate_synthetic_isimip_data
from src.budyko_model.isimip_attribution import ISIMIPAttribution

# 生成测试数据
data = generate_synthetic_isimip_data(range(1960, 2017), Q_o_base=200.0, Q_n_base=250.0, n_models=9, add_noise=True)

# 修改数据
for model in data['model_names']:
    data['isimip_data']['obsclim_1901soc'].loc[
        data['isimip_data']['obsclim_1901soc']['year'] >= 1986,
        model
    ] = 50.0

attribution = ISIMIPAttribution(data['obs_data'], data['isimip_data'])
attribution.set_periods()
results = attribution.run_attribution()

print(f"贡献率之和: {results['contribution_sum']:.1f}%")
print(f"偏离量: {abs(results['contribution_sum'] - 100):.1f}%")
print(f"触发警告阈值: 15%")
