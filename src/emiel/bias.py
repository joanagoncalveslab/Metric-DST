from src.emiel.datagen.conceptshift import shifter as con_shift
from src.emiel.datagen.conceptshift import selector as con_sel
from src.emiel.datagen.covshift import selector as cov_sel

def call_bias(x, y, bias, bias_params):
    '''
        'class_imbalance', 'reduce_samples', 'bias_2features', 'bias_most_important_feature', 'bias_multi_features', 'bias_balanced_multi_features'
    '''
    
    x_2bias, y_2bias = x.copy(), y.copy() 
    if 'concept' in bias['name']: 
        shifter = con_shift.Shifter(n_domains=bias_params['n_domains'], rot=0, trans=bias_params['trans'], scale=0)
        selector = con_sel.DomainSelector(n_global=bias_params['n_global'], n_source=bias_params['n_source'], n_target=bias_params['n_target'],
                            n_domains_source=1, n_domains_target=1)
        x_2bias, y_2bias, domain = shifter.shift(x_2bias, y_2bias)
        X_unlabeled, y_unlabeled, X_b_train, y_b_train, X_test, y_test = selector.select(x_2bias, y_2bias, domain)
    elif 'covariate' in bias['name']: 
        selector = cov_sel.FeatureSelector(n_global=bias_params['n_global'], n_source=bias_params['n_source'], n_target=bias_params['n_target'],
                               source_scale=bias_params['source_scale'], target_scale=bias_params['target_scale'], bias_dist=bias_params['bias_dist'])
        X_unlabeled, y_unlabeled, X_b_train, y_b_train, X_test, y_test = selector.select(x_2bias, y_2bias)
    return X_unlabeled, y_unlabeled, X_b_train, y_b_train, X_test, y_test
