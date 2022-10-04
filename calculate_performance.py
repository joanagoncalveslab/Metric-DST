import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

performance = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-exclude-NARS2-noMiner-contrastive/performance.csv", index_col=0)
test_performance = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-exclude-NARS2-noMiner-contrastive/performance-testset.csv", index_col=0)

performance = performance[(performance['epoch']==performance['epoch'].max())]
performance = performance.iloc[:, 3:]
test_performance = test_performance.iloc[:, 1:]
print(pd.concat([test_performance.mean(), test_performance.std()], axis=1, keys=['mean', 'std']).T)
print(pd.concat([performance.mean(), performance.std()], axis=1, keys=['mean', 'std']).T)

performance = performance.melt(var_name='cols', value_name='vals')
performance['set']='validation set in last epoch without gene'
test_performance = test_performance.melt(var_name='cols', value_name='vals')
test_performance['set']='test set with gene'

combined = pd.concat([performance, test_performance])

sns.set_theme(style="whitegrid")
g = sns.catplot(data=combined, kind='bar' , x='cols', y='vals', hue='set', errorbar='sd', palette='dark', alpha=.6, height=6)
g.despine(left=True)
g.set_axis_labels("", 'value')
g.legend.set_title("")

# for container in g.ax.containers:
#     g.ax.bar_label(container)

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(g.ax)
plt.show()