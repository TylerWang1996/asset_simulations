import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def boxplot_monthly_returns(data_df: pd.DataFrame, start_date: str = None, end_date: str = None, title: str = None,
                            figsize: tuple = (8, 6)) -> plt:
    """
    Generate a boxplot of monthly returns for financial data.

    Parameters:
    data_df (pd.DataFrame): The input financial data.
    start_date (str): The starting date to filter data (inclusive).
    end_date (str): The ending date to filter data (inclusive).
    title (str): The title of the plot.
    figsize (tuple): The dimensions of the plot in inches. Default is (8, 6).

    Returns:
    plt: The matplotlib plot object.
    """

    # Reset index to use 'Date' as a column
    data_df = data_df.reset_index()

    # Filter data based on start and end dates
    if start_date and end_date:
        mask = (data_df['Date'] >= start_date) & (data_df['Date'] <= end_date)
        data_df = data_df.loc[mask].reset_index(drop=True)

    # Melt data to create long-form DataFrame
    data_melt_df = pd.melt(data_df, id_vars=['Date'])
    data_melt_df = data_melt_df.rename(columns={'variable': 'Asset', 'value': 'Monthly Returns'})

    # Set Seaborn style to white
    sns.set_style("white")

    # Generate boxplot with Seaborn
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x='Asset', y='Monthly Returns', data=data_melt_df, ax=ax)
    ax.set(xlabel=None)

    # Format y-axis with percentage formatter
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Set title if specified
    if title:
        ax.set_title(title, fontsize=16)

    # Return plot object
    return plt


def summary_stats_table(data_df: pd.DataFrame, start_date: str = None, end_date: str = None, title: str = None) -> pd.DataFrame:
    """
    Generate a summary statistics table for financial data.

    Parameters:
    data_df (pd.DataFrame): The input financial data.
    start_date (str): The starting date to filter data (inclusive).
    end_date (str): The ending date to filter data (inclusive).
    title (str): The title of the table.

    Returns:
    pd.DataFrame: The summary statistics table.
    """

    # Reset index to use 'Date' as index
    if start_date and end_date:
        data_df = data_df.reset_index()
        mask = (data_df['Date'] >= start_date) & (data_df['Date'] <= end_date)
        data_df = data_df.loc[mask].reset_index(drop=True)
        data_df = data_df.set_index('Date')

    # Compute summary statistics with pandas
    agg_stats = data_df.agg(['mean', 'std', 'median', 'min', 'max', 'skew', 'kurtosis'])

    # Rename index labels
    agg_stats = agg_stats.rename(index={'mean': 'Mean', 'median': 'Median', 'min': 'Min', 'max': 'Max', 'skew': 'Skew', 'kurtosis': 'Kurtosis'})

    # Rename 'std' to 'Volatility'
    agg_stats = agg_stats.rename(index={'std': 'Volatility'})

    # Format numeric values
    decimals_rows = agg_stats.index.isin(['Skew', 'Kurtosis'])
    agg_stats[~decimals_rows] = agg_stats[~decimals_rows].applymap(lambda x: '{:.2f}%'.format(x * 100))
    agg_stats[decimals_rows] = agg_stats[decimals_rows].applymap(lambda x: '{:.2f}'.format(x))

    # Set title if specified
    if title:
        agg_stats = agg_stats.rename(columns={'': title})

    # Return summary statistics table
    return agg_stats


def plot_corr_matrix(data_df: pd.DataFrame, start_date: str = None, end_date: str = None, title: str = None,
                     figsize: tuple = (10, 7)) -> plt:
    """
    Generate a correlation matrix plot for financial data.

    Parameters:
    data_df (pd.DataFrame): The input financial data.
    start_date (str): The starting date to filter data (inclusive).
    end_date (str): The ending date to filter data (inclusive).
    title (str): The title of the plot.
    figsize (tuple): The dimensions of the plot in inches. Default is (10, 7).

    Returns:
    plt: The matplotlib plot object.
    """

    # Reset index to use 'Date' as a column
    if start_date and end_date:
        data_df = data_df.reset_index()
        mask = (data_df['Date'] >= start_date) & (data_df['Date'] <= end_date)
        data_df = data_df.loc[mask].reset_index(drop=True)

    # Compute correlation matrix with pandas
    corrmat = data_df.corr().round(2)

    # Set plot size and generate heatmap with Seaborn
    plt.figure(figsize=figsize)
    sns.heatmap(corrmat, vmax=1, annot=True, fmt=".2f", linewidths=.5)
    plt.xticks(rotation=30, horizontalalignment='right')

    # Set title if specified
    if title:
        plt.title(title, fontsize=16)

    # Return plot object
    return plt
