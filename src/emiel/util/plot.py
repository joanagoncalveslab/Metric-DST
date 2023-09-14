import pandas as pd

from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

FORMAT_PERC = FuncFormatter(lambda y, _: '{:.0%}'.format(y))

DISPLAY_NAMES = {
    "s-only": "$S_{only}$",
    "g-only": "Global-only",
    "t-only": "$T_{only}$",
    "s->g": "$S\\to G$",
    "s->t": "$S\\to T$"
}


def plot_target_acc_box(results: pd.DataFrame, title: str, save: str = None) -> None:
    """
    Plot for each model the target accuracy boxplot.
    :param results: dataframe in the format returned by batch_load_eval
    :param title: plot title
    :param save: path to save figure. If None, shows figure directly.
    """
    cols = ['s-only', 's->g', 's->t', 't-only']

    acc = _process_target_acc(results)
    acc = acc[cols]
    acc = acc.rename(columns=DISPLAY_NAMES)

    plt.figure(dpi=300, figsize=(3.5, 3.5))
    acc.boxplot(ax=plt.gca(), showfliers=False)
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.ylim(bottom=0.4, top=1)
    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()


def plot_adaptation(results: pd.DataFrame, title: str = None) -> None:
    """
    For adapt-to-global, and adapt-to-target,
    plot the percentage that they reach between source-only and target-only.
    :param results: dataframe in the format returned by batch_load_eval
    :param title
    """
    acc = _process_target_acc(results)
    adaptation_g = (acc['s->g'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
    adaptation_t = (acc['s->t'] - acc['s-only']) / (acc['t-only'] - acc['s-only'])
    data = pd.DataFrame({DISPLAY_NAMES['s->g']: adaptation_g,
                         DISPLAY_NAMES['s->t']: adaptation_t})
    data.boxplot(showfliers=False)
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Adaptation (%)")
    plt.tight_layout()
    plt.show()


def plot_relative_adaptation(df: pd.DataFrame, title: str = None) -> None:
    acc = _process_target_acc(df)
    adaptation_g_rel_t = (acc['s->g'] - acc['s-only']) / (acc['s->t'] - acc['s-only'])
    data2 = pd.DataFrame({DISPLAY_NAMES['s->g']: adaptation_g_rel_t})
    data2.boxplot(showfliers=False)
    plt.gca().yaxis.set_major_formatter(FORMAT_PERC)

    plt.title(title)
    plt.ylabel("Relative Adaptation(%)")
    plt.tight_layout()
    plt.show()


def print_acc_stats(df: pd.DataFrame, alpha=0.002):
    """Not a plotting function, but similar processing steps. Outputs to terminal."""
    acc = _process_target_acc(df)
    cols = ['s-only', 's->g', 's->t', 't-only']
    for col in cols:
        series = acc[col]
        _, p = ttest_ind(acc['s-only'], series, equal_var=False)
        print(f'{col:8} median: {series.median():.3f}, '
              f'mean: {series.mean():.3f}, '
              f'std: {series.std():.3f}, '
              f'p={p:.3f} '
              + ('(SIGN)' if p < alpha else ''))


def _process_target_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only target acc columns and rename to 'x->y' or 'x-only' format.
    Sorts columns by median."""
    acc = df.loc[:, df.columns.str.contains('acc-on-t')]
    acc = acc.rename(columns=lambda col: col.split('metrics.')[1].split('-acc-on-t')[0])
    sorted_columns = acc.median().sort_values()
    return acc[sorted_columns.index]



