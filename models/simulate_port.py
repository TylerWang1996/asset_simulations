import pandas as pd
import numpy as np


def portfolio_randomizer(asset_ranges, iterations=100000, seed=None):
    """
    Generate random portfolio weights that satisfy constraints on asset ranges.

    Parameters:
    -----------
    asset_ranges : dict
        A dictionary of assets and their allowed ranges. Each asset is a key in the dictionary, and its value is a list
        containing two floats: the lower and upper bounds for the asset's weight in the portfolio.
    iterations : int, optional (default=100000)
        The number of random portfolio weights to generate.
    seed : int or None, optional (default=None)
        The seed to use for the random number generator. If None, a random seed will be used.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the generated portfolio weights, with each row representing one portfolio and each column
        representing one asset.
    """

    # Set random seed if provided
    if seed:
        np.random.seed(seed)

    # Get list of assets
    assets = list(asset_ranges.keys())

    # Generate random portfolio weights using a Dirichlet distribution
    num_assets = len(assets)
    randomizer = np.random.dirichlet(np.ones(num_assets), size=iterations)
    df_ports = pd.DataFrame(randomizer, columns=assets)

    # Apply constraints to portfolio weights
    df_bound_ports = df_ports.copy()
    for asset in assets:
        lower_bound, upper_bound = asset_ranges[asset]
        df_bound_ports = df_bound_ports[(df_bound_ports[asset] >= lower_bound) & (df_bound_ports[asset] <= upper_bound)]
    df_bound_ports.reset_index(drop=True, inplace=True)

    return df_bound_ports


if __name__ == "__main__":
    # Example usage
    asset_ranges = {'US3M T-Bill': [0.0, 0.1], '7-10Y US Treasury': [0.15, 0.35],
                    'IG Credit Bonds': [0.15, 0.30], 'HY Credit Bonds': [0.10, 0.30],
                    'Equities': [0.10, 0.25], 'Commodities': [0.0, 0.10]}

    sample_ranges = {'Equity': [0.2, 0.35], 'Bonds': [0.4, 0.5], 'Cash': [0.1, 0.2]}

    rand_port = portfolio_randomizer(sample_ranges, iterations=100000, seed=1234)
    print(rand_port.dtypes)
    print(rand_port.sum(axis=1))

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    # Creating dataset
    z = rand_port['Equity']
    x = rand_port['Bonds']
    y = rand_port['Cash']

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x, y, z, color="green")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_zlim([0.0, 1.0])
    plt.title("Asset Weights Scatter Plot")

    # Axis labels
    ax.set_xlabel('Rates', fontweight='bold')
    ax.set_ylabel('Credit', fontweight='bold')
    ax.set_zlabel('Equities', fontweight='bold')

    # show plot
    plt.show()