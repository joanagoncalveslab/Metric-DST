import os,sys
project_path = '.'
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'dssl4sl':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
#from __future__ import print_function
from sklearn import datasets as sk_ds
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from src import config
import pandas as pd
import numpy as np
import pickle

R_STATE = 123
#pd.read_excel('tmp.xlsx', index_col=0)
params = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "subsample": 0.9, "subsample_freq": 1,
    #"min_child_weight":0.01,
    "reg_lambda": 5,
    "class_weight": 'balanced',
    "random_state": R_STATE, "verbose": -1, "n_jobs":-1,
}


def split_dataset(X, y, train_ratio=0.20, test_ratio=0.20, unknown_ratio=None, r_seed=123):
    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=1 - train_ratio, random_state=r_seed)
    if unknown_ratio is None:
        return x_train, x_test, y_train, y_test
    else:
        x_unk, x_test, y_unk, y_test = train_test_split(x_test, y_test, shuffle=True, stratify=y_test,
                                                        test_size=test_ratio / (test_ratio + unknown_ratio),
                                                        random_state=r_seed)
        return {'x_train': x_train, 'y_train': y_train,
                'x_unk': x_unk, 'y_unk': y_unk,
                'x_test': x_test, 'y_test': y_test}


def tryclassify(x_train, x_test, y_train, y_test):
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=124)
    from sklearn.ensemble import RandomForestClassifier
    #clf = RandomForestClassifier(random_state=124, n_jobs=-1)
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(boosting_type="rf", **params)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(score)


# Links: https://www.muratkoklu.com/datasets/
def load_fire(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'muratkoklu' / 'Acoustic_Extinguisher_Fire_Dataset' / 'Acoustic_Extinguisher_Fire_Dataset.xlsx'
    data = pd.read_excel(train_url, engine='openpyxl')

    y = data['STATUS'].values
    data['FUEL'] = data['FUEL'].replace(data['FUEL'].unique(), np.arange(len(data['FUEL'].unique())))
    X = data.drop(columns=['STATUS']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# Links: https://www.muratkoklu.com/datasets/
def load_pistachio(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'muratkoklu' / 'Pistachio_Dataset' / 'Pistachio_16_Features_Dataset' / 'Pistachio_16_Features_Dataset.xlsx'
    data = pd.read_excel(train_url, engine='openpyxl')

    y = data['Class'].replace(data['Class'].unique(), np.arange(len(data['Class'].unique()))).values
    X = data.drop(columns=['Class']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# Links: https://www.muratkoklu.com/datasets/
def load_pumpkin(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'muratkoklu' / 'Pumpkin_Seeds_Dataset' / 'Pumpkin_Seeds_Dataset.xlsx'
    data = pd.read_excel(train_url, engine='openpyxl')

    y = data['Class'].replace(data['Class'].unique(), np.arange(len(data['Class'].unique()))).values
    X = data.drop(columns=['Class']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# Links: https://www.muratkoklu.com/datasets/
def load_raisin(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'muratkoklu' / 'Raisin_Dataset' / 'Raisin_Dataset.xlsx'
    data = pd.read_excel(train_url, engine='openpyxl')

    y = data['Class'].replace(data['Class'].unique(), np.arange(len(data['Class'].unique()))).values
    X = data.drop(columns=['Class']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# Links: https://www.muratkoklu.com/datasets/
def load_rice(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'muratkoklu' / 'Rice_Dataset_Commeo_and_Osmancik' / 'Rice_Cammeo_Osmancik.xlsx'
    data = pd.read_excel(train_url, engine='openpyxl')

    y = data['Class'].replace(data['Class'].unique(), np.arange(len(data['Class'].unique()))).values
    X = data.drop(columns=['Class']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# https://archive.ics.uci.edu/ml/datasets/Yeast
def load_abalone(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'UCI' / 'abalone.data'
    names_url = config.DATA_DIR / 'datasets' / 'UCI' / 'abalone.names'
    data = pd.read_csv(train_url, sep=',', engine='python', header=None)
    data['label']=0
    data.loc[data[8]<15, 'label']=1
    y = data['label'].values
    X = data.drop(columns=[0, 8, 'label']).values

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

# https://archive.ics.uci.edu/ml/datasets/Yeast
def load_yeast(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'UCI' / 'yeast.data'
    names_url = config.DATA_DIR / 'datasets' / 'UCI' / 'yeast.names'
    data = pd.read_csv(train_url, sep='\s+', engine='python', header=None)
    y = data[data.shape[1]-1].replace(["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", 'POX', 'ERL'],
                         [0, 1, 2, 3, 3, 3, 4, 4, 4, 4]).values
    X = data.drop(columns=[0, data.shape[1]-1]).values
    X = X[y<2]
    y = y[y<2]

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train


def load_spam(feature_names=False, test=False):
    train_url = config.DATA_DIR / 'datasets' / 'UCI' / 'spambase.data'
    names_url = config.DATA_DIR / 'datasets' / 'UCI' / 'spambase.names'
    data = pd.read_csv(train_url, sep=',', engine='python', na_values="?", header=None).values
    y = data[:,-1].astype('int')
    X = data[:,:-1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

def load_adult(feature_names=False, test=False):
    features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

    # Change these to local file if available
    train_url = config.DATA_DIR / 'datasets' / 'UCI' / 'adult.data'
    test_url = config.DATA_DIR / 'datasets' / 'UCI' / 'adult.test'
    original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*',
                                 engine='python', na_values="?")
    original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',
                                engine='python', na_values="?", skiprows=1)
    num_train = len(original_train)
    original = pd.concat([original_train, original_test])
    labels = original['Target'].replace('<=50K', 0).replace('<=50K.', 0).replace('>50K', 1).replace('>50K.', 1).values
    original = original.drop(columns=['Target', 'Education'])
    data = pd.get_dummies(original)

    x_train = data[:num_train].values
    x_test = data[num_train:].values
    y_train = labels[:num_train]
    y_test = labels[num_train:]

    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.values, labels
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train

def load_drug(drug='CX-5461', feature_names=False, test=False):
    folder = config.DATA_DIR / 'datasets' / 'drug'
    drug_feature_loc = folder / f'binarized_{drug}_features.csv'
    with open(drug_feature_loc, 'rb') as handle:
        feature_data = pickle.load(handle)
    X = feature_data['data']
    y = feature_data['label']
    X_names = feature_data['feature_names']

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, X_names
    else:
        return x_train, y_train


def load_oxaliplatin(feature_names=False, test=False):
    folder = config.DATA_DIR / 'datasets' / 'drug'
    oxa_loc = folder / 'oxaliplatin_60.csv'
    if os.path.exists(oxa_loc):
        oxaliplatin = pd.read_csv(oxa_loc)
    else:
        resp_loc = folder / 'GDSC2_fitted_dose_response_25Feb20.xlsx'
        a = pd.read_excel(resp_loc, engine='openpyxl')
        oxaliplatin = a[a['DRUG_NAME'] == 'Oxaliplatin']
        oxaliplatin['label'] = -1
        oxaliplatin.loc[oxaliplatin['AUC'] <= oxaliplatin['AUC'].quantile(0.40), 'label'] = 0
        oxaliplatin.loc[oxaliplatin['AUC'] >= oxaliplatin['AUC'].quantile(0.60), 'label'] = 1
        oxaliplatin.to_csv(oxa_loc, index=None)

    data = oxaliplatin[['COSMIC_ID', 'label']]
    data = data[data['label'] != -1]
    expr_loc = folder / 'Cell_line_RMA_proc_basalExp_drug.txt'
    expr = pd.read_csv(expr_loc, sep='\t')
    expr = expr.drop(columns=['GENE_title'])
    expr = expr.set_index('GENE_SYMBOLS')
    expr = expr.T
    expr5000 = expr.iloc[:, expr.var().argsort()[::-1].values[:5000]]
    expr5000.index = expr5000.index.str.split('.').str[1]
    expr_dict = expr5000.T.to_dict('list')
    sample_names = data['COSMIC_ID'].astype('str')
    X_c = sample_names.map(expr_dict).values
    X_list = X_c.tolist()
    good_indices = [row_id for row_id, row in enumerate(X_list) if type(row) == list]
    X = [X_list[ind] for ind in good_indices]
    X = np.array(X)
    y = data['label'].values[good_indices]
    X_names = sample_names.values[good_indices]

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return X, y
    if feature_names:
        return x_train, y_train, X_names
    else:
        return x_train, y_train

#'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension','radius error', 'texture error', 'perimeter error', 'area error','smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error','fractal dimension error', 'worst radius', 'worst texture','worst perimeter', 'worst area', 'worst smoothness','worst compactness', 'worst concavity', 'worst concave points','worst symmetry', 'worst fractal dimension'
def load_breast_cancer(feature_names=False, test=False):
    data = sk_ds.load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, shuffle=True, stratify=data.target,
                                                        test_size=0.2, random_state=R_STATE)
    # x_train, y_train = data.data, data.target
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.data, data.target
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train


def load_diabetes(feature_names=False, test=False):
    data = sk_ds.load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, shuffle=True, stratify=data.target,
                                                        test_size=0.2, random_state=R_STATE)
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.data, data.target
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train


def load_mnist(feature_names=False, test=False):
    data = sk_ds.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, shuffle=True, stratify=data.target,
                                                        test_size=0.2, random_state=R_STATE)
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.data, data.target
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train


def load_iris(feature_names=False, test=False):
    data = sk_ds.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, shuffle=True, stratify=data.target,
                                                        test_size=0.2, random_state=R_STATE)
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.data, data.target
    if feature_names:
        return x_train, y_train, data.feature_names
    else:
        return x_train, y_train


def load_mushroom(feature_names=False, test=False):
    data_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'mushrooms.csv'
    data = pd.read_csv(data_loc)
    y = data["class"].replace(["e", "p"], [1, 0]).values
    # oh_data = data.drop(columns=['class']).astype('category')
    data = data.drop(columns=['class'])
    oh_data = pd.get_dummies(data)
    x_train, x_test, y_train, y_test = train_test_split(oh_data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return oh_data.values, y
    if feature_names:
        return x_train, y_train, oh_data.columns
    else:
        return x_train, y_train


# https://www.kaggle.com/ronitf/heart-disease-uci #302 samples
def load_heart(feature_names=False, test=False):
    data_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'heart.csv'
    data = pd.read_csv(data_loc)
    y = data["target"].values
    # oh_data = data.drop(columns=['class']).astype('category')
    data = data.drop(columns=['target'])
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.values, y
    if feature_names:
        return x_train, y_train, data.columns
    else:
        return x_train, y_train


# https://www.kaggle.com/rupakroy/credit-data?select=credit_data.csv #2000 samples
def load_credit(feature_names=False, test=False):
    data_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'credit_data.csv'
    data = pd.read_csv(data_loc)
    y = data["default"].values
    # oh_data = data.drop(columns=['class']).astype('category')
    data = data.drop(columns=['clientid', 'default'])
    imputed = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data.values)
    x_train, x_test, y_train, y_test = train_test_split(imputed, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return imputed, y
    if feature_names:
        return x_train, y_train, data.columns
    else:
        return x_train, y_train


# https://www.kaggle.com/yasserh/wine-quality-dataset?select=WineQT.csv
def load_wine(feature_names=False, test=False):
    data_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'WineQT.csv'
    data = pd.read_csv(data_loc)
    y = data["quality"].replace([3, 4, 5, 6, 7, 8], [0, 0, 0, 1, 1, 1]).values
    # oh_data = data.drop(columns=['class']).astype('category')
    data = data.drop(columns=['quality', 'Id'])
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.values, y
    if feature_names:
        return x_train, y_train, data.columns
    else:
        return x_train, y_train


# https://archive.ics.uci.edu/ml/datasets/wine+quality
def load_wine_uci(feature_names=False, test=False):
    data1_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'winequality-red.csv'
    data2_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'winequality-white.csv'
    data1 = pd.read_csv(data1_loc, sep=';')
    data2 = pd.read_csv(data2_loc, sep=';')
    data = pd.concat([data1, data2])
    data = data[data['quality'].isin([5, 6, 7])]
    y = data['quality'].values
    data = data.drop(columns=['quality'])
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.values, y
    if feature_names:
        return x_train, y_train, data.columns
    else:
        return x_train, y_train

    # https://archive.ics.uci.edu/ml/datasets/wine+quality


def load_wine_uci2(feature_names=False, test=False):
    data1_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'winequality-red.csv'
    data2_loc = config.DATA_DIR / 'datasets' / 'UCI' / 'winequality-white.csv'
    data1 = pd.read_csv(data1_loc, sep=';')
    data1['class'] = 0
    data2 = pd.read_csv(data2_loc, sep=';')
    data2['class'] = 1
    data = pd.concat([data1, data2])
    data = data.drop(columns=['quality'])
    y = data['class'].values
    data = data.drop(columns=['class'])
    data = data.dropna()
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)
    # data = sk_ds.load_breast_cancer()
    if test:
        return x_train, y_train, x_test, y_test
    else:
        return data.values, y
    if feature_names:
        return x_train, y_train, data.columns
    else:
        return x_train, y_train


def load_cifar(class_size=10, flattened=True, test=False):
    from keras import datasets as keras_ds
    # The data, split between train and test sets:
    if class_size == 100:
        (x_train, y_train), (x_test, y_test) = keras_ds.cifar100.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras_ds.cifar10.load_data()
    x_train_f = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    x_test_f = x_test.reshape(x_test.shape[0], np.prod(x_train.shape[1:]))
    if flattened:
        if test is not None and test == True:
            return x_train, y_train.flatten(), x_test, y_test.flatten()
        else:
            return (x_train_f, y_train.flatten())  # , (x_test_f, y_test.flatten())
    else:
        return (x_train, y_train)  # , (x_test, y_test)
    
def load_cifar_torch():
    #import torch
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                        shuffle=True, num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                        shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_ppi(id=9606, drop_samples=False, test=False):
    file_loc = f'{id}.protein.links.full.v11.5.txt.gz_64_0.9_features.pkl'
    if id == 9606:
        file_loc = f'{id}.protein.links.full.v11.0.txt_64_0.9_features.pkl'
    file_loc = config.DATA_DIR / 'graphs' / 'STRING' / file_loc
    data = pd.read_pickle(file_loc)
    y = data['class'].values
    samples = data[['protein1', 'protein2']]
    if drop_samples:
        data = data.drop(columns=['class', 'protein1', 'protein2'])
    else:
        data = data.drop(columns=['class'])
    data = data.dropna()
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    if test is not None and test == True:
        return x_train, y_train, x_test, y_test
    return x_train, y_train


def load_cora(chosen_class='Neural_Networks', drop_samples=False, test=False):
    file_loc = f'cora_{chosen_class}_features.pkl.gz'
    file_loc = config.DATA_DIR / 'graphs' / 'linqs' / 'cora' / file_loc
    data = pd.read_pickle(file_loc)
    y = data['class'].values
    samples = data[['ent1', 'ent2']]
    if drop_samples:
        data = data.drop(columns=['class', 'ent1', 'ent2'])
    else:
        data = data.drop(columns=['class'])
    data = data.dropna()
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    if test is not None and test == True:
        return x_train, y_train, x_test, y_test
    return x_train, y_train


def load_webkb(source='washington', chosen_class='student', pca=None, drop_samples=False, test=False):
    if pca is None:
        file_loc = f'{source}_{chosen_class}_features.pkl.gz'
    else:
        file_loc = f'{source}_{chosen_class}_{pca}_features.pkl.gz'
    file_loc = config.DATA_DIR / 'graphs' / 'linqs' / 'webkb' / file_loc
    data = pd.read_pickle(file_loc)
    y = data['class'].values
    samples = data[['ent1', 'ent2']]
    if drop_samples:
        data = data.drop(columns=['class', 'ent1', 'ent2'])
    else:
        data = data.drop(columns=['class'])
    data = data.dropna()
    x_train, x_test, y_train, y_test = train_test_split(data.values, y, shuffle=True, stratify=y,
                                                        test_size=0.2, random_state=R_STATE)

    if test is not None and test == True:
        return x_train, y_train, x_test, y_test
    return x_train, y_train


def load_dataset(dataset_name, *args, **kwargs):
    return globals()[f'load_{dataset_name}'](*args, **kwargs)


if __name__ == '__main__':
    print('Main load dataset')
    for ds in ['breast_cancer', 'wine_uci2','mushroom', 'mnist', 'fire', 'spam', 'adult', 'rice', 'raisin', 'pistachio', 'pumpkin']:
        x_train, y_train, x_test, y_test = load_dataset(ds, test=True)
        print(f'{ds}: {x_train.shape[0]+x_test.shape[0]} | {x_train.shape[1]}')
        print(y_test)