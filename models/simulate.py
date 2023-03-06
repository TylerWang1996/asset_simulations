import pandas as pd
import numpy as np


class TimeSeriesSimulate:
    """Simulates geometric Brownian motion for multiple assets.

    Attributes:
        tickers (list of str): Tickers of the assets.
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

        # calculate the standard deviations of the variables
        std_devs = np.sqrt(np.diag(covar))

        # divide the covariance matrix by the outer product of the standard deviations
        corr_matrix = covar / np.outer(std_devs, std_devs)

        chol = np.linalg.cholesky(corr_matrix)

        return chol

    def simulate_gbm(self, mu=None, sigma=None):
        """Simulates geometric Brownian motion for one asset.

        Args:
            mu (float or None): Mean return of the asset. If None, use the first element of self.means.
            sigma (float or None): Standard deviation of the asset. If None, use the square root of the first diagonal element of self.covar.

        Returns:
            numpy.ndarray: Simulated prices of the asset.
        """
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
        """Simulates geometric Brownian motion for multiple assets.

        Returns:
            list of pandas.DataFrame: Simulated prices of each asset.
        """
        n_assets = len(self.means)

        d_sim = {}
        for i in range(n_assets):
            ticker = self.tickers[i]
            mu = self.means[i]
            sigma = np.sqrt(self.covar[i][i])
            sim = self.simulate_gbm(mu, sigma)
            df_sim = pd.DataFrame(sim)
            d_sim[ticker] = df_sim

        dfs_sim = []
        chol = self.cholesky_decomp(self.covar)
        for i in range(self.sims):
            df_sim = pd.DataFrame(columns=self.tickers)
            for col in df_sim:
                df_sim[col] = d_sim[col][i]
            df_sim = np.matmul(chol, df_sim.T).T
            df_sim.set_axis(self.tickers, axis=1, inplace=True)
            dfs_sim.append(df_sim)

        return dfs_sim


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
