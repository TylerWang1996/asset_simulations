import pandas as pd
import numpy as np


def portfolio_randomizer(d_port, iterations=100000, seed=None):
    if seed:
        np.random.seed(seed)

    l_assets = d_port.keys()

    num_assets = len(l_assets)
    randomizer = np.random.dirichlet(np.ones(num_assets), size=iterations)
    df_ports = pd.DataFrame(randomizer)
    df_ports.columns = l_assets
    df_bound_ports = df_ports.copy()

    for i in l_assets:
        bound = d_port[i]
        lower = bound[0]
        upper = bound[1]
        df_bound_ports = df_bound_ports[(df_bound_ports[i] >= lower) & (df_bound_ports[i] <= upper)]

    df_bound_ports.reset_index(drop=True, inplace=True)

    return df_bound_ports


if __name__ == "__main__":
    d_port = {'A': [0.0, 0.1], 'B': [0.15, 0.30], 'C': [0.0, 0.05], 'D': [0.15, 0.30], 'E': [0.10, 0.30],
              'F': [0.10, 0.25], 'G': [0.0, 0.10]}

    rand_port = portfolio_randomizer(d_port, seed=1234)