import pandas as pd
import numpy as np


class TimeSeriesSimulate:
    def __init__(self, means, covar, years, steps_per_year, seed=None, sims=100):
        """
        A class for simulating time series using Geometric Brownian Motion.

        Parameters:
            means (list): A list of means for the Geometric Brownian Motion process.
            covar (list of lists): A covariance matrix for the Geometric Brownian Motion process.
            years (int): The number of years to simulate.
            steps_per_year (int): The number of steps to take per year.
            seed (int): (optional) The random seed to use for generating the normal variates.
            sims (int): (optional) The number of simulations to run.
        """
        self.means = means
        self.covar = covar
        self.years = years
        self.steps_per_year = steps_per_year
        self.seed = seed
        self.sims = sims

    def cholesky_decomp(self):
        """
        Performs a Cholesky decomposition of the covariance matrix.

        Returns:
            list of lists: The lower triangular Cholesky decomposition of the covariance matrix.
        """
        # TODO: Implement cholesky_decomp method
        pass

    def simulate_gbm(self, mu=None, sigma=None):
        """
        Simulates Geometric Brownian Motion using the Euler-Maruyama method.

        Parameters:
            mu (float): The drift coefficient. If not specified, the first mean in the list of means is used.
            sigma (float): The volatility coefficient. If not specified, the square root of the first element
                           of the diagonal of the covariance matrix is used.

        Returns:
            np.ndarray: A numpy array of simulated stock prices of shape (n+1, M).
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

        # multiply through by initial stock prices
        St = np.multiply.outer(self.means, St[-1, :])

        return St

if __name__ == "__main__":
    means = [0.1, 0.05, 0.15]
    covar = [[0.2, 0.1, 0.15],
             [0.1, 0.3, 0.2],
             [0.15, 0.2, 0.4]]
    years = 5
    steps_per_year = 12
    seed = 123
    sims = 100

    # Create an instance of the TimeSeriesSimulate class
    ts = TimeSeriesSimulate(means, covar, years, steps_per_year, seed=seed, sims=sims)

    # Simulate Geometric Brownian Motion
    gbm = ts.simulate_gbm()

    # Display the results
    print(gbm)
