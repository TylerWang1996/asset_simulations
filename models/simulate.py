import pandas as pd
import numpy as np


class TimeSeriesSimulate:
    def __init__(self, tickers, means, covar, years, steps_per_year, seed=None, sims=100):

        self.tickers = tickers
        self.means = means
        self.covar = covar
        self.years = years
        self.steps_per_year = steps_per_year
        self.seed = seed
        self.sims = sims

    def cholesky_decomp(self, covar):

        chol = np.linalg.cholesky(covar)

        return chol

    def simulate_gbm(self, mu=None, sigma=None):

        # extract parameters from class attributes
        n = self.steps_per_year * self.years
        T = self.years
        M = self.sims
        S0 = 100
        seed = self.seed

        # extract mu and sigma from class attributes or use defaults
        if mu is None:
            mu = self.means[0]
        if sigma is None:
            sigma = np.sqrt(self.covar[0][0])

        # Set the random seed
        if seed is not None:
            np.random.seed(seed)

        # calculate each time step
        dt = T / n

        # simulation using numpy arrays
        St = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
        )

        # include array of 1's
        St = np.vstack([np.ones(M), St])

        # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
        St = S0 * St.cumprod(axis=0)

        return St

    def agg_simulation(self):

        n_assets = len(self.means)

        d_sim = {}
        i = 0
        while i < n_assets:
            ticker = self.tickers[i]
            mu = self.means[i]
            sigma = np.sqrt(self.covar[i][i])
            sim = self.simulate_gbm(mu, sigma)
            df_sim = pd.DataFrame(sim)
            d_sim[ticker] = df_sim
            i += 1

        dfs_sim = []
        i = 0
        chol = self.cholesky_decomp(self.covar)
        while i < self.sims:
            df_sim = pd.DataFrame(columns=self.tickers)
            for col in df_sim:
                df_sim[col] = d_sim[col][i]
            df_sim = np.matmul(chol, df_sim.T).T
            df_sim.set_axis(self.tickers, axis=1, inplace=True)
            dfs_sim.append(df_sim)
            i += 1

        return dfs_sim


def simulate_stock_prices_returns(prices_returns_df, years, steps_per_year, seed, sims):
    # Calculate mean and covariance matrix
    mean_returns = prices_returns_df.mean().to_numpy()
    cov_matrix = prices_returns_df.cov().to_numpy()
    print(mean_returns)
    print(prices_returns_df.mean())

    # Extract tickers from columns of prices_returns_df
    tickers = list(prices_returns_df.columns)
    print(tickers)

    # Create an instance of the TimeSeriesSimulate class
    ts = TimeSeriesSimulate(tickers, mean_returns, cov_matrix, years, steps_per_year, seed=seed, sims=sims)

    # Simulate Geometric Brownian Motion
    gbm = ts.agg_simulation()

    return gbm


if __name__ == "__main__":
    # create a random stock prices returns dataframe with three columns (tickers) and 100 rows (dates)
    np.random.seed(123)
    prices_returns_df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    prices_returns_df = prices_returns_df / 100

    # set input parameters
    years = 5
    steps_per_year = 12
    seed = 123
    sims = 2

    # call the simulate_stock_prices_returns function
    gbm = simulate_stock_prices_returns(prices_returns_df, years, steps_per_year, seed, sims)

    # display the results
    print(gbm)
