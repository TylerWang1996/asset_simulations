import pandas as pd
import numpy as np

daterange = pd.date_range('1970-01-01','2023-03-01', freq='MS').strftime("%Y-%m-%d").tolist()
periods = len(daterange)

df = pd.DataFrame(daterange, columns=['Date'])

l_assets = list('ABCEDFG')
n_assets = len(l_assets)

np.random.seed(12345)
seeds = np.random.randint(low=0, high=10000, size=n_assets)

for i in range(n_assets):
    seed = seeds[i]
    asset = l_assets[i]
    np.random.seed(seed)
    mu = np.random.uniform(0, 0.1)
    ret = np.random.normal(mu, mu, periods)
    df[asset] = ret

df = df.set_index('Date')
df.to_csv('sample_data.csv')