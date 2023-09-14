import numpy as np
#from src.andrei.helper_methods import generate_continuous_dataset
from sklearn.datasets import make_moons, make_classification
from sklearn import preprocessing
import matplotlib.pyplot as plt



def three_unbalanced_clusters(a_size=None, b_size=None):
    # Set sample size for the two classes
    if a_size is None or b_size is None:
        a_size = [1100, 400]
        b_size = [1500]

    # Set seed to ensure experiment repeatability
    np.random.seed(5)

    # Class means
    a_means = [np.array([-7.0, 6.0]), np.array([7.0, 6.0])]
    b_means = [np.array([0.0, 0.0])]

    # Class covariances
    a_cov = [np.array([[4.5, 0], [0, 4.5]]), np.array([[2, 0], [0, 2]])]
    b_cov = [np.array([[15, 0], [0, 15]])]

    x_set, y_set = generate_continuous_dataset(a_means, a_cov, a_size, b_means, b_cov, b_size,
                                               file_name="datasets/three_unbalanced_clusters.csv")
    return x_set, y_set


def overlapped_linear_clusters(a_size=None, b_size=None):
    # Set sample size for the two classes
    if a_size is None or b_size is None:
        a_size = [1500]
        b_size = [1500]

    # Set seed to ensure experiment repeatability
    np.random.seed(5)

    # Class means
    a_means = [np.array([-4.0, -4.0])]
    b_means = [np.array([-20.0, 2.0])]

    # Class covariances
    a_cov = [np.array([[8, 7], [7, 8]])]
    b_cov = [np.array([[80, -60], [-60, 80]])]

    x_set, y_set = generate_continuous_dataset(a_means, a_cov, a_size, b_means, b_cov, b_size,
                                               file_name="datasets/overlapped_linear_clusters.csv")
    return x_set, y_set


def rotated_moons(n_samples=None, noise=(0.05, 0.35)):
    # Determine sample size for the two classes
    if n_samples is None:
        n_samples = (1500, 1500)

    x_set, y_set = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return x_set, y_set


def custom_mdimension(n_samples, n_features, random_state, flip_y=0.1, class_sep=1.0):
    x_scaled, y_set = make_classification(n_samples=n_samples,
                                       n_features=50,
                                       n_informative=int(n_features * 0.7),
                                       n_redundant=int(n_features * 0.3),
                                       n_repeated=0,
                                       n_classes=2,
                                       n_clusters_per_class=4,
                                       flip_y=flip_y,
                                       class_sep=class_sep,
                                       random_state=random_state)
    #x_scaled = preprocessing.StandardScaler().fit_transform(x_scaled)
    return x_scaled, y_set

def multidimensional_set(n_samples, n_features, random_state):
    if n_samples is None:
        n_samples = int((n_features * n_features * 5) * 2.5)
    x_set, y_set = make_classification(n_samples=n_samples,
                                       n_features=n_features,
                                       n_informative=int(n_features * 0.7),
                                       n_redundant=int(n_features * 0.3),
                                       n_repeated=0,
                                       n_classes=2,
                                       n_clusters_per_class=4,
                                       flip_y=0,
                                       class_sep=1.0,
                                       random_state=random_state)
    x_scaled = preprocessing.StandardScaler().fit_transform(x_set)
    return x_scaled, y_set

if __name__ == '__main__':
    n_samples, n_features, random_state, flip_y, class_sep = 
    x, y = custom_mdimension(500, 50, 0, 0.1, 1.0)
    if x.shape[1]==2:
        emb = x.copy()
    else:
        import umap
        reducer = umap.UMAP()
        emb = reducer.fit_transform(x)
    c_dict = {0: '#994455', 1: '#004488'}
    plt.scatter(emb[:, 0], emb[:, 1], marker="o", c=c_dict, s=25, edgecolors='none')
    plt.savefig('deneme.png')
    #plt.show()