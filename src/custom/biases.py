from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from statistics import mean
import scipy.stats
from sklearn.decomposition import PCA


def delta(x_train, y_train, size, b, delta1, delta2):
    np.random.seed(123)
    class0_len = sum(y_train==0)
    class1_len = sum(y_train==1)
    class0_ids = np.arange(len(y_train))[y_train==0]
    class1_ids = np.arange(len(y_train))[y_train==1]
    x_train_0 = x_train[class0_ids]
    x_train_1 = x_train[class1_ids]
    
    selection_probs_0 =  np.exp(-b * (np.abs(x_train_0[:, 0] - delta1[0]) + np.abs(x_train_0[:, 1] - delta1[1])))
    selection_probs_1 =  np.exp(-b * (np.abs(x_train_1[:, 0] - delta2[0]) + np.abs(x_train_1[:, 1] - delta2[1])))
    
    selected_class0_ids = np.random.choice(class0_ids, size, replace=False, p=selection_probs_0 / selection_probs_0.sum())
    selected_class1_ids = np.random.choice(class1_ids, size, replace=False, p=selection_probs_1 / selection_probs_1.sum())
    
    selected_ids = np.concatenate([selected_class0_ids, selected_class1_ids], axis=None)
    return np.sort(selected_ids)
    '''
    joined = np.append(x_train, y_train.reshape((len(y_train), 1)), axis=1)
    x_train_0 = joined[joined[:, 2] == 0, 0:2]
    x_train_1 = joined[joined[:, 2] == 1, 0:2]
    y_train_0 = joined[joined[:, 2] == 0, 2:3].flatten()
    y_train_1 = joined[joined[:, 2] == 1, 2:3].flatten()

    sample_bias_0 = np.exp(-b * (np.abs(x_train_0[:, 0] - delta_1[0]) + np.abs(x_train_0[:, 1] - delta_1[1])))
    biased_index_0 = np.random.choice(range(len(x_train_0)), size, replace=False,
                                      p=sample_bias_0 / sample_bias_0.sum())
    x_biased_0 = x_train_0[biased_index_0]
    y_biased_0 = y_train_0[biased_index_0]

    sample_bias_1 = np.exp(-b * (np.abs(x_train_1[:, 0] - delta_2[0]) + np.abs(x_train_1[:, 1] - delta_2[1])))
    biased_index_1 = np.random.choice(range(len(x_train_1)), size, replace=False,
                                      p=sample_bias_1 / sample_bias_1.sum())
    x_biased_1 = x_train_1[biased_index_1]
    y_biased_1 = y_train_1[biased_index_1]

    x_biased = np.append(x_biased_0, x_biased_1, axis=0)
    y_biased = np.append(y_biased_0, y_biased_1, axis=0)

    return x_biased, y_biased
    '''

def call_bias(x, y, name, **params):
    '''
        'class_imbalance', 'reduce_samples', 'bias_2features', 'bias_most_important_feature', 'bias_multi_features', 'bias_balanced_multi_features'
    '''
    print(f'Bias: {name}')
    if params is None:
        print('Params is none')
    elif name == 'delta':
        if params is None:
            sel_ids = delta(x, y, 50, 1.5, 1.5, (1, 0.5), (0,0))
        else:
            sel_ids = delta(x, y, **params)
    
    return sel_ids