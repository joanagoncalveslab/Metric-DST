import pandas as pd


def get_corr_mat(df: pd.DataFrame):
    """Get correlation matrix for all numeric columns that are non-constant."""
    df = df.select_dtypes(include=['number'])
    std_values = df.std()
    selected_columns = std_values[std_values != 0].index
    return df[selected_columns].corr()


def print_highest_corr(corr: pd.DataFrame, num: int = 5):
    """Print the N highest correlation combinations to the console."""
    correlation_series = corr.stack()
    sorted_correlations = correlation_series.sort_values(ascending=False)

    count = 0
    for (var1, var2), correlation in sorted_correlations.items():
        if var1 != var2:
            print(f"Variables: {var1}, {var2} | Correlation: {correlation:.3f}")
            count += 1
        if count >= num:
            break
