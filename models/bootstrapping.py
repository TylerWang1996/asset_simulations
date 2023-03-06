import numpy as np
import pandas as pd
from typing import Union, List


class BlockBootstrap:
    """
    Bootstrap using blocks of the same length with optional wrap-around
    Parameters
    ----------
    block_size : int
        Size of block to use
    data : pd.DataFrame
        DataFrame containing the data to bootstrap. The DataFrame should have
        datetime index and stock tickers as columns.
    seed : {{int, numpy.random.Generator, numpy.random.RandomState}}, optional
        Seed to use to ensure reproducible results. If an int, passes the
        value to value to ``numpy.random.default_rng``. If None, a fresh
        Generator is constructed with system-provided entropy.
    wrap_around : bool, optional
        Whether or not to use wrap-around when resampling. Default is True.
    Attributes
    ----------
    block_size : int
        Size of block used for bootstrapping
    data : pd.DataFrame
        DataFrame containing the data to bootstrap.
    seed : {{int, numpy.random.Generator, numpy.random.RandomState}}, optional
        Seed used to ensure reproducible results. Passed to
        ``numpy.random.default_rng``.
    wrap_around : bool
        Whether or not to use wrap-around when resampling.
    """

    def __init__(
        self,
        block_size: int,
        data: pd.DataFrame,
        seed: Union[int, np.random.Generator, np.random.RandomState, None] = None,
        wrap_around: bool = True,
    ) -> None:
        """
        Initializes a BlockBootstrap instance.
        """
        self.block_size = block_size
        self.data = data
        self.seed = seed
        self.wrap_around = wrap_around

    def bootstrap(self, bs_seed: int) -> pd.DataFrame:
        """
        Generates a new resampled data set using block bootstrapping.

        Parameters
        ----------
        bs_seed : int
            Seed used to generate the bootstrap sample.

        Returns
        -------
        pd.DataFrame
            DataFrame of the resampled data set, with the same datetime index
            and stock tickers as columns as the original data.
        """
        rng = np.random.default_rng(bs_seed)
        num_blocks = self.data.shape[0] // self.block_size
        if num_blocks * self.block_size < self.data.shape[0]:
            num_blocks += 1

        # Generate random indices for the resampling
        if self.wrap_around:
            # Indices wrap around to the start of the data
            indices = rng.integers(0, self.data.shape[0], size=num_blocks)
            indices = indices[:, None] + np.arange(self.block_size)
            indices = indices.flatten()
            indices %= self.data.shape[0] # Enables wrap-around
        else:
            # Indices do not wrap around to the start of the data
            max_index = self.data.shape[0] - self.block_size + 1
            indices = rng.integers(0, max_index, size=num_blocks)
            indices = indices[:, None] + np.arange(self.block_size)
            indices = indices.flatten()

        # Resample the data using the random indices
        resampled_data = self.data.iloc[indices, :]
        return resampled_data

    def generate_bootstraps(self, num_replications: int = 1) -> List[pd.DataFrame]:
        """
        Generates new resampled data sets using block bootstrapping.

        Parameters
        ----------
        num_replications : int, optional
            Number of resampled data sets to generate. Default is 1.

        Returns
        -------
        List[pd.DataFrame]
            List of resampled data sets, where each resampled data set is a
            DataFrame with the same datetime index and stock tickers as columns
            as the original data.
        """
        rng = np.random.default_rng(self.seed)
        # Generate seeds for each bootstrap sample
        num_seeds = num_replications
        bootstrap_seeds = rng.integers(0, 2 ** 32, size=num_seeds)

        bootstrap_samples = []
        for bs_seed in bootstrap_seeds:
            sample = self.bootstrap(bs_seed)
            bootstrap_samples.append(sample)

        return bootstrap_samples


if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    tickers = ["AAPL", "GOOGL", "TSLA"]
    data = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=tickers)
    seed = 12345

    # Test for proper shape of output
    bs = BlockBootstrap(5, data, seed=seed, wrap_around=False)
    bs_sample = bs.generate_bootstraps(1)

    print(bs_sample)
