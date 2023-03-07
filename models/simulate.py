import pandas as pd
import numpy as np


class TimeSeriesSimulate:
    """Simulates geometric Brownian motion for multiple assets.

    Attributes:
        tickers (list): Tickers of the assets.
        means (numpy.ndarray): Mean returns for each asset.
        covar (numpy.ndarray): Covariance matrix of returns for each asset.
        years (int): Number of years to simulate.
        steps_per_year (int): Number of time steps per year to simulate.
        seed (int or None): Seed for the random number generator.
        sims (int): Number of simulations to run.
    """
    def __init__(self, tickers, means, covar, years, steps_per_year, seed=None, sims=100):
        self.tickers = tickers
        self.means = means
        self.covar = covar
        self.years = years
        self.steps_per_year = steps_per_year
        self.seed = seed
        self.sims = sims

    def cholesky_decomp(self, covar):
        """Performs Cholesky decomposition on the covariance matrix.

        Args:
            covar (numpy.ndarray): Covariance matrix.

        Returns:
            numpy.ndarray: Lower triangular matrix from Cholesky decomposition.
        """
        v = np.sqrt(np.diag(covar))
        outer_v = np.outer(v, v)
        correlation = covar / outer_v
        correlation[covar == 0] = 0

        chol = np.linalg.cholesky(correlation)

        return chol

    def simulate_gbm(self, mu=None, sigma=None, seed=None):
        """Simulates geometric Brownian motion for one asset.

        Args:
            mu (float or None): Mean return of the asset. If None, use the first element of self.means.
            sigma (float or None): Standard deviation of the asset. If None, use the square root of the first diagonal element of self.covar.
            seed (int or None): Seed for the random number generator.

        Returns:
            numpy.ndarray: Simulated prices of the asset.
        """
        # extract parameters from class attributes
        n = self.steps_per_year * self.years
        T = self.years
        M = self.sims

        # Set the random seed
        np.random.seed(seed)

        # calculate each time step
        dt = T / n

        if mu is None:
            mu = self.means[0]
        if sigma is None:
            sigma = np.sqrt(self.covar[0, 0])

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

    def agg_simulation(self):
        """Simulates geometric Brownian motion for multiple assets.

        Returns:
            list of pandas.DataFrame: Simulated prices of each asset.
        """
        n_assets = len(self.means)
        np.random.seed(self.seed)
        seeds = np.random.randint(low=0, high=10000, size=n_assets)

        d_sim = {}
        for i in range(n_assets):
            ticker = self.tickers[i]
            mu = self.means[i]
            sigma = np.sqrt(self.covar[i][i])
            sim = self.simulate_gbm(mu, sigma, seeds[i])
            df_sim = pd.DataFrame(sim)
            d_sim[ticker] = df_sim

        dfs_sim = []
        for i in range(self.sims):
            df_sim = pd.DataFrame(columns=self.tickers)
            for col in df_sim:
                df_sim[col] = d_sim[col][i]
            dfs_sim.append(df_sim)

        #Use cholesky decomposotion to add in correlation effect
        chol = self.cholesky_decomp(self.covar)
        dfs_series = []
        for df in dfs_sim:
            copy_df = df.copy()
            ticker_cols = copy_df.columns
            correl_df = pd.DataFrame(np.matmul(chol, copy_df.T)).T
            ticker_names_dict = {old_name: new_name for old_name, new_name in zip(correl_df.columns, ticker_cols)}
            correl_df = correl_df.rename(columns=ticker_names_dict)
            series_df = correl_df + 1
            series_df = series_df.cumprod(axis=0)
            dfs_series.append(series_df)

        return dfs_series


def simulate_stock_prices_returns(prices_returns_df, years, steps_per_year, seed, sims):
    """Simulates geometric Brownian motion for a given set of asset prices.

    Args:
        prices_returns_df (pandas.DataFrame): DataFrame containing the asset prices.
        years (int): Number of years to simulate.
        steps_per_year (int): Number of time steps per year to simulate.
        seed (int): Seed for the random number generator.
        sims (int): Number of simulations to run.

    Returns:
        list of pandas.DataFrame: Simulated prices of each asset.
    """
    # Calculate mean and covariance matrix
    mean_returns = prices_returns_df.mean().to_numpy()

    cov_matrix = prices_returns_df.cov().to_numpy()

    # Extract tickers from columns of prices_returns_df
    tickers = list(prices_returns_df.columns)

    # Create an instance of the TimeSeriesSimulate class
    ts = TimeSeriesSimulate(tickers, mean_returns, cov_matrix, years, steps_per_year, seed=seed, sims=sims)

    # Simulate Geometric Brownian Motion
    gbm = ts.agg_simulation()

    return gbm


if __name__ == "__main__":
    # create a random stock prices returns dataframe with three columns (tickers) and 100 rows (dates)
    prices_returns_df = pd.read_csv('sample_data.csv', index_col=0)

    # set input parameters
    years = 10
    steps_per_year = 252
    seed = 12345
    sims = 2

    # call the simulate_stock_prices_returns function
    gbm = simulate_stock_prices_returns(prices_returns_df, years, steps_per_year, seed=seed, sims=sims)