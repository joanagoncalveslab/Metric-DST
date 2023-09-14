import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_accuracy_curve(x_values, x_label, acc_log_reg, acc_iw1=None, acc_iw2=None, acc_optimal=None, acc_optimal2=None):
    if acc_optimal is not None:
        plt.plot(x_values, acc_optimal, "k", label="optimal (domain)")
    if acc_optimal2 is not None:
        plt.plot(x_values, acc_optimal2, "#f3a200", label="optimal (unbiased)")
    plt.plot(x_values, acc_log_reg, "b", label="unweighted")
    if acc_iw1 is not None:
        plt.plot(x_values, acc_iw1, "r", label="KMM (\u03B3 = 1)")
    if acc_iw2 is not None:
        plt.plot(x_values, acc_iw2, "g", label="KLIEP (\u03B3 = 10)")

    plt.xlabel(x_label, fontsize=17)
    plt.ylabel("accuracy", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(x_values, x_values)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()


def plot_da_acc_percentages(x_values, iw_vec, labels_vec, title, colors):
    for i in range(len(iw_vec)):
        plt.plot(x_values, iw_vec[i], c=colors[i], label=labels_vec[i])

    plt.xlabel("class 2 proportion (%)", fontsize=18)
    plt.ylabel("domain adaptation (%)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(x_values, x_values)
    plt.legend(loc="upper left", fontsize=16)
    plt.title(title)
    plt.show()


def plot_dataset(features, labels, weights_iw=None, weights_log_reg=None):
    if weights_iw is None:
        weights_iw = np.array([50] * len(labels))
    if weights_log_reg is None:
        weights_log_reg = np.array([1, 1])

    colors = np.array(['cornflowerblue', 'darkorange'])

    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], s=weights_iw[labels == 0] * weights_log_reg[0],
                edgecolors='black', linewidths=0.6, c=colors[0], label="class 1")
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], s=weights_iw[labels == 1] * weights_log_reg[1],
                edgecolors='black', linewidths=0.6, c=colors[1], label="class 2")

    plt.axis('equal')
    plt.xlabel('feature 1', fontsize=20)
    plt.ylabel('feature 2', fontsize=20)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=14)
    plt.show()


def plot_probability_density_function(x_set, y_set, title=""):
    joined = np.append(x_set, y_set.reshape((len(y_set), 1)), axis=1)
    x_0 = joined[joined[:, 2] == 0, 0:2]
    x_1 = joined[joined[:, 2] == 1, 0:2]
    x_0_f1 = x_0[:, 0]
    x_0_f2 = x_0[:, 1]
    x_1_f1 = x_1[:, 0]
    x_1_f2 = x_1[:, 1]
    sns.set_style("white")
    sns.kdeplot(x=x_1_f1, y=x_1_f2, fill=True, cmap="Reds", alpha=0.85, bw_adjust=0.8, cbar=True)
    sns.kdeplot(x=x_0_f1, y=x_0_f2, fill=True, cmap="Blues", alpha=0.55, bw_adjust=0.8, cbar=True)

    plt.xlabel('feature 1', fontsize=20)
    plt.ylabel('feature 2', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.title(title, fontsize=20)
    plt.show()


def plot_bias_sample_size_heatmap(data_acc_iw, optimal_acc):
    subtract_optimal = lambda i: optimal_acc - i
    vectorized = np.vectorize(subtract_optimal)
    data_processed = vectorized(data_acc_iw)

    x_labels = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    y_labels = [4, 3, 2, 1, 0, "- 1", "- 2", "- 3", "- 4"]

    fig = plt.figure(figsize=(9, 9))
    sns.set(font_scale=1.7)
    heat_map = sns.heatmap(data_processed, linewidth=1, xticklabels=x_labels, yticklabels=y_labels, vmin=0, annot=False)
    plt.xlabel("sample size (%)", fontsize=25)
    plt.ylabel("bias impact (n)", fontsize=25)
    plt.title("KLIEP (\u03B3 = 0.1)", fontsize=28)
    plt.show()


def compute_marginal_std_from_heatmap(data_acc_iw, optimal_acc):
    subtract_optimal = lambda i: optimal_acc - i
    vectorized = np.vectorize(subtract_optimal)
    data_processed = vectorized(data_acc_iw)

    # axis = 0 (project on x_axis; sample size)
    # axis = 1 (project on y-axis; bias impact)
    proj_0 = np.sum(data_processed, axis=0)
    std0 = np.std(proj_0)
    proj_1 = np.sum(data_processed, axis=1)
    std1 = np.std(proj_1)
    return std0, std1


def generate_continuous_dataset(means0, covs0, sizes0, means1, covs1, sizes1, store=False, file_name="data.csv"):
    # Generate the random samples
    samples0 = np.random.multivariate_normal(means0[0], covs0[0], sizes0[0])
    samples1 = np.random.multivariate_normal(means1[0], covs1[0], sizes1[0])

    for i, _ in enumerate(sizes0[1:]):
        new_samples = np.random.multivariate_normal(means0[i + 1], covs0[i + 1], sizes0[i + 1])
        samples0 = np.r_[samples0, new_samples]

    for i, _ in enumerate(sizes1[1:]):
        new_samples = np.random.multivariate_normal(means1[i + 1], covs1[i + 1], sizes1[i + 1])
        samples1 = np.r_[samples1, new_samples]

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int8)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int8)]

    # Construct the dataset and split into features and training sets
    dataset = np.r_[class0, class1]
    features, labels = np.hsplit(dataset, [2])
    labels = labels.flatten()

    # Plot the resulting distribution only if it contains two features + target
    if dataset.shape[1] == 3:
        plot_dataset(features, labels)

    # Store in a csv file
    if store is True:
        dataframe = pd.DataFrame(data=dataset, columns=['feature1', 'feature2', 'target'])
        dataframe.to_csv(file_name, index=False, float_format='%1.4f')

    return features, labels
