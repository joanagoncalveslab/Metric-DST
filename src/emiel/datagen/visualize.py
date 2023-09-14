"""Visualization functions for datasets created with the datagen module"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import plotly.express as px
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def visualize_shift2d(xg, yg, xs, ys, xt, yt, title=None):
    """Plot the positions of the source, global and target sets, with markers for binary labels.
    Uses matplotlib."""

    _, ax1 = plt.subplots(1, 1, dpi=200, figsize=(4, 4))
    if title:
        ax1.set_title(title)
    for x, y, label, c in [
        (xg, yg, 'global', 'green'),
        (xs, ys, 'source', 'blue'),
        (xt, yt, 'target', 'red')]:
        pos = x[y == 1]
        neg = x[y == 0]

        ax1.scatter(pos[:, 0], pos[:, 1], s=10, label=label, c=c, marker='o')
        ax1.scatter(neg[:, 0], neg[:, 1], s=10, c=c, marker='x')
    ax1.legend(loc="lower right")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    plt.show()


def visualize_shift2d_px(xg, yg, xs, ys, xt, yt):
    """Plot the positions of the source, global and target sets, with markers for binary labels.
    Uses plotly for interactive notebooks."""

    data = {
        'feature1': np.concatenate([xg[:, 0], xs[:, 0], xt[:, 0]]),
        'feature2': np.concatenate([xg[:, 1], xs[:, 1], xt[:, 1]]),
        'label': np.concatenate([yg, ys, yt]),
        'domain': ['global'] * len(xg) + ['source'] * len(xs) + ['target'] * len(xt)}
    df = pd.DataFrame(data)
    symbol_mapping = {1: 'circle', 0: 'x'}
    color_mapping = {'global': '#00CC96', 'source': '#636EFA', 'target': '#EF553B'}
    fig = px.scatter(df, x='feature1', y='feature2',
                     color='domain', color_discrete_map=color_mapping,
                     symbol='label', symbol_map=symbol_mapping)
    fig.show()


def visualize_shift3d_px(xg, yg, xs, ys, xt, yt):
    """Plot the positions of the source, global and target sets, with markers for binary labels.
    Uses plotly for interactive notebooks."""

    data = {
        'feature1': np.concatenate([xg[:, 0], xs[:, 0], xt[:, 0]]),
        'feature2': np.concatenate([xg[:, 1], xs[:, 1], xt[:, 1]]),
        'feature3': np.concatenate([xg[:, 2], xs[:, 2], xt[:, 2]]),
        'label': np.concatenate([yg, ys, yt]),
        'domain': ['global'] * len(xg) + ['source'] * len(xs) + ['target'] * len(xt)}
    df = pd.DataFrame(data)
    symbol_mapping = {1: 'circle', 0: 'x'}
    color_mapping = {'global': 'green', 'source': 'blue', 'target': 'orange'}
    fig = px.scatter_3d(df, x='feature1', y='feature2', z='feature3',
                        color='domain', color_discrete_map=color_mapping,
                        symbol='label', symbol_map=symbol_mapping)
    fig.update_traces(marker=dict(size=2))
    fig.show()


def visualize_decision_boundary2d(xs, ys, xt, yt, model, name=None):
    """Given a trained model, plots the: labeled source data, the decision boundary,
     the position of the target data, and computes the accuracy.
     Optionally add a name to the plot title.
     Taken mostly from: https://adapt-python.github.io/adapt/examples/Two_moons.html"""
    yt_pred = model.predict(xt)
    acc = accuracy_score(yt, yt_pred > 0.5)

    x_min, y_min = np.min([xs.min(0), xt.min(0)], 0)
    x_max, y_max = np.max([xs.max(0), xt.max(0)], 0)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    x_grid_ = np.stack([x_grid.ravel(), y_grid.ravel()], -1)
    yp_grid = model.predict(x_grid_).reshape(100, 100)

    x_pca = np.concatenate((model.encoder_.predict(xs),
                            model.encoder_.predict(xt)))
    x_pca = PCA(2).fit_transform(x_pca)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title("Input space")
    ax1.contourf(x_grid, y_grid, yp_grid, cmap=cm.RdBu, alpha=0.6)
    ax1.scatter(xs[ys == 0, 0], xs[ys == 0, 1],
                label="source", edgecolors='k', c="red")
    ax1.scatter(xs[ys == 1, 0], xs[ys == 1, 1],
                label="source", edgecolors='k', c="blue")
    ax1.scatter(xt[:, 0], xt[:, 1], label="target", edgecolors='k', c="black")
    ax1.legend()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')

    ax2.set_title("PCA encoded space")
    ax2.scatter(x_pca[:len(xs), 0][ys == 0], x_pca[:len(xs), 1][ys == 0],
                label="source", edgecolors='k', c="red")
    ax2.scatter(x_pca[:len(xs), 0][ys == 1], x_pca[:len(xs), 1][ys == 1],
                label="source", edgecolors='k', c="blue")
    ax2.scatter(x_pca[len(xs):, 0], x_pca[len(xs):, 1],
                label="target", edgecolors='k', c="black")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.tick_params(direction='in')

    title = f"Target Acc : {acc:.3f}"
    if name:
        title = f"{name} - {title}"
    fig.suptitle(title)
    plt.show()
