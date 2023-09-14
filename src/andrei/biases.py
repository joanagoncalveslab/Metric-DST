from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from statistics import mean
import scipy.stats
from sklearn.decomposition import PCA


def bias_labels(x_train, y_train, i, n_class_partitions):
    joined = np.append(x_train, y_train.reshape((len(y_train), 1)), axis=1)
    x_train_0 = joined[joined[:, 2] == 0, 0:2]
    x_train_1 = joined[joined[:, 2] == 1, 0:2]
    y_train_0 = joined[joined[:, 2] == 0, 2:3].flatten()
    y_train_1 = joined[joined[:, 2] == 1, 2:3].flatten()

    x_train_0, x_discard, y_train_0, y_discard = train_test_split(x_train_0, y_train_0,
                                                                  train_size=(50 - i * 50 / n_class_partitions) / 100,
                                                                  random_state=7)
    x_train_1, x_discard, y_train_1, y_discard = train_test_split(x_train_1, y_train_1,
                                                                  train_size=(50 + i * 50 / n_class_partitions) / 100,
                                                                  random_state=9)
    x_train_biased = np.append(x_train_0, x_train_1, axis=0)
    y_train_biased = np.append(y_train_0, y_train_1, axis=0)

    return x_train_biased, y_train_biased


def bias_features_and_reduce_samples(x_train, y_train, i, n_class_partitions, b_1, b_2, delta_1, delta_2):
    np.random.seed(123)

    joined = np.append(x_train, y_train.reshape((len(y_train), 1)), axis=1)
    x_train_0 = joined[joined[:, 2] == 0, 0:2]
    x_train_1 = joined[joined[:, 2] == 1, 0:2]
    y_train_0 = joined[joined[:, 2] == 0, 2:3].flatten()
    y_train_1 = joined[joined[:, 2] == 1, 2:3].flatten()

    sample_bias_0 = np.exp(-b_1 * (np.abs(x_train_0[:, 0] - 1.0) + np.abs(x_train_0[:, 1] - 0.5)))
    biased_index_0 = np.random.choice(range(len(x_train_0)), int(len(x_train_0) * (n_class_partitions - i) /
                                                                 n_class_partitions), replace=False,
                                      p=sample_bias_0 / sample_bias_0.sum())
    x_biased_0 = x_train_0[biased_index_0]
    y_biased_0 = y_train_0[biased_index_0]

    sample_bias_1 = np.exp(-b_2 * (np.abs(x_train_1[:, 0] + 0) + np.abs(x_train_1[:, 1] + 0)))
    biased_index_1 = np.random.choice(range(len(x_train_1)), int(len(x_train_1) * (n_class_partitions - i) /
                                                                 n_class_partitions), replace=False,
                                      p=sample_bias_1 / sample_bias_1.sum())
    x_biased_1 = x_train_1[biased_index_1]
    y_biased_1 = y_train_1[biased_index_1]

    x_biased = np.append(x_biased_0, x_biased_1, axis=0)
    y_biased = np.append(y_biased_0, y_biased_1, axis=0)

    return x_biased, y_biased


def reduce_samples_only(x_train, y_train, i, n_class_partitions):
    x_reduced, x_disc, y_reduced, y_disc = train_test_split(x_train, y_train, train_size=((n_class_partitions - i)
                                                            / n_class_partitions) - 0.00001, random_state=12)
    return x_reduced, y_reduced


def __index_most_important_feature(x_set, y_set):
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, stratify=y_set, random_state=42)
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    return np.argmax(importances)


def bias_most_important_feature(x_train, y_train, size, a, b):
    np.random.seed(123)

    joined = np.append(x_train, y_train.reshape((len(y_train), 1)), axis=1)
    n = len(joined[0])
    x_train_0 = joined[joined[:, n-1] == 0, 0:(n-1)]
    x_train_1 = joined[joined[:, n-1] == 1, 0:(n-1)]
    y_train_0 = joined[joined[:, n-1] == 0, (n-1):n].flatten()
    y_train_1 = joined[joined[:, n-1] == 1, (n-1):n].flatten()

    index_feature = __index_most_important_feature(x_train, y_train)
    feature_val = x_train[:, index_feature].flatten()
    m = min(feature_val)
    m_hat = mean(feature_val)
    mean_normal = m + (m_hat - m)/a
    std_normal = np.sqrt((m_hat - m)/b)

    feature_val_0 = x_train_0[:, index_feature].flatten()
    feature_val_1 = x_train_1[:, index_feature].flatten()
    sample_bias_0 = scipy.stats.norm(loc=mean_normal, scale=std_normal).pdf(feature_val_0)
    sample_bias_1 = scipy.stats.norm(loc=mean_normal, scale=std_normal).pdf(feature_val_1)

    biased_index_0 = np.random.choice(range(len(x_train_0)), size, replace=False, p=sample_bias_0/sum(sample_bias_0))
    x_biased_0 = x_train_0[biased_index_0]
    y_biased_0 = y_train_0[biased_index_0]

    biased_index_1 = np.random.choice(range(len(x_train_1)), size, replace=False,
                                      p=sample_bias_1 / sum(sample_bias_1))
    x_biased_1 = x_train_1[biased_index_1]
    y_biased_1 = y_train_1[biased_index_1]

    x_biased = np.append(x_biased_0, x_biased_1, axis=0)
    y_biased = np.append(y_biased_0, y_biased_1, axis=0)

    return x_biased, y_biased


def bias_all_features_and_target_PCA(x_train, y_train, size, a, b):
    np.random.seed(35)

    principal_comp = PCA(n_components=1).fit_transform(x_train).flatten()
    m = min(principal_comp)
    m_hat = mean(principal_comp)
    mean_normal = m + (m_hat - m)/a
    std_normal = np.sqrt((m_hat - m)/b)

    sample_bias = scipy.stats.norm(loc=mean_normal, scale=std_normal).pdf(principal_comp)

    biased_index = np.random.choice(range(len(x_train)), size, replace=False, p=sample_bias/sum(sample_bias))
    x_biased = x_train[biased_index]
    y_biased = y_train[biased_index]

    return x_biased, y_biased


def bias_all_features_excl_target_PCA(x_train, y_train, size, a, b):
    np.random.seed(35)

    joined = np.append(x_train, y_train.reshape((len(y_train), 1)), axis=1)
    n = len(joined[0])
    x_train_0 = joined[joined[:, n-1] == 0, 0:(n-1)]
    x_train_1 = joined[joined[:, n-1] == 1, 0:(n-1)]
    y_train_0 = joined[joined[:, n-1] == 0, (n-1):n].flatten()
    y_train_1 = joined[joined[:, n-1] == 1, (n-1):n].flatten()

    principal_comp = PCA(n_components=1).fit_transform(x_train)
    m = min(principal_comp.flatten())
    m_hat = mean(principal_comp.flatten())
    mean_normal = m + (m_hat - m)/a
    std_normal = np.sqrt((m_hat - m)/b)

    joined_pc = np.append(principal_comp, y_train.reshape((len(y_train), 1)), axis=1)
    pc_val_0 = joined_pc[joined_pc[:, 1] == 0, 0].flatten()
    pc_val_1 = joined_pc[joined_pc[:, 1] == 1, 0].flatten()

    sample_bias_0 = scipy.stats.norm(loc=mean_normal, scale=std_normal).pdf(pc_val_0)
    sample_bias_1 = scipy.stats.norm(loc=mean_normal, scale=std_normal).pdf(pc_val_1)

    biased_index_0 = np.random.choice(range(len(x_train_0)), size, replace=False,
                                      p=sample_bias_0/sum(sample_bias_0))
    x_biased_0 = x_train_0[biased_index_0]
    y_biased_0 = y_train_0[biased_index_0]

    biased_index_1 = np.random.choice(range(len(x_train_1)), size, replace=False,
                                      p=sample_bias_1 / sum(sample_bias_1))
    x_biased_1 = x_train_1[biased_index_1]
    y_biased_1 = y_train_1[biased_index_1]

    x_biased = np.append(x_biased_0, x_biased_1, axis=0)
    y_biased = np.append(y_biased_0, y_biased_1, axis=0)

    return x_biased, y_biased

def call_bias(x, y, name, **params):
    '''
        'class_imbalance', 'reduce_samples', 'bias_2features', 'bias_most_important_feature', 'bias_multi_features', 'bias_balanced_multi_features'
    '''
    print(f'Bias: {name}')
    if params is None:
        print('Params is none')
    if name == 'class_imbalance':
        if params is None:
            x_biased, y_biased = bias_labels(x, y, 5, 10)
        else:
            x_biased, y_biased = bias_labels(x, y, params['i'], params['n_class_partitions'])
    elif name == 'bias_2features':
        if params is None:
            x_biased, y_biased = bias_features_and_reduce_samples(x, y, 5, 10, 1.5, 1.5, (1, 0.5), (0,0))
        else:
            x_biased, y_biased = bias_features_and_reduce_samples(x, y, **params)
    elif name == 'bias_most_important_feature':
        if params is None:
            x_biased, y_biased = bias_most_important_feature(x, y, 50, 3, 4)
        else:
            x_biased, y_biased = bias_most_important_feature(x, y, **params)
    elif name == 'bias_multi_features':
        if params is None:
            x_biased, y_biased = bias_all_features_and_target_PCA(x, y, 50, 3, 4)
        else:
            x_biased, y_biased = bias_all_features_and_target_PCA(x, y, **params)
    elif name == 'bias_balanced_multi_features':
        if params is None:
            x_biased, y_biased = bias_all_features_excl_target_PCA(x, y, 50, 3, 4)
        else:
            x_biased, y_biased = bias_all_features_excl_target_PCA(x, y, **params)
    
    return x_biased, y_biased 