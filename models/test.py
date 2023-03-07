import pandas as pd
import numpy as np


def simulate_gbm(mu, sigma, seed):
    """Simulates geometric Brownian motion for one asset.

    Args:
        mu (float or None): Mean return of the asset. If None, use the first element of self.means.
        sigma (float or None): Standard deviation of the asset. If None, use the square root of the first diagonal element of self.covar.

    Returns:
        numpy.ndarray: Simulated prices of the asset.
    """
    # extract parameters from class attributes
    n = 252 * 5
    T = 5
    M = 100
    S0 = 100

    # Set the random seed

    # calculate each time step
    dt = T / n

    # simulation using numpy arrays
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    )

    # include array of 1's
    St = np.vstack([np.ones(M), St])

    #get the returns only
    St = St - 1

    return St


if __name__ == "__main__":
    sim1 = simulate_gbm(0.1, 0.1)
    sim2 = simulate_gbm(0.25, 0.25)

    sim1_df = pd.DataFrame(sim1)
    sim2_df = pd.DataFrame(sim2)
    d_sim = {'sim1': sim1_df, 'sim2': sim2_df}

    tickers = ['sim1', 'sim2']
    means = [0.1, 0.25]
    sigmas = [0.1, 0.25]

    for i in range(2):
        ticker = tickers[i]
        mu = means[i]
        sigma = sigmas[i]
        sim = simulate_gbm(mu, sigma)
        df_sim = pd.DataFrame(sim)
        d_sim[ticker] = df_sim

    dfs_sim=[]
    for i in range(100):
        df_sim = pd.DataFrame(columns=['sim1', 'sim2'])
        for col in df_sim:
            df_sim[col] = d_sim[col][i]
        dfs_sim.append(df_sim)

    print(dfs_sim[0].corr())