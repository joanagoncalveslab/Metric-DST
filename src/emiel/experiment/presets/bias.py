from datagen.covshift.builder import CovShiftBuilder
from datagen.covshift.selector import FeatureSelector
from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter

N_INITIAL = 10000
N_GLOBAL = 1000
N_SOURCE = 1000
N_TARGET = 1000

CONCEPT_N_DOMAINS = 4
CONCEPT_TRANS_WEAK = 2
CONCEPT_TRANS_STRONG = 4

COV_SCALE_WEAK = 1
COV_SCALE_STRONG = 0.75
COV_DIST_WEAK = 2
COV_DIST_STRONG = 3


_init_classification = dict(n_samples=N_INITIAL,
                            n_features=5,
                            n_informative=3, n_redundant=2, n_repeated=0,
                            n_clusters_per_class=4)


def concept_weak() -> ConceptShiftDataBuilder:
    shifter = Shifter(n_domains=CONCEPT_N_DOMAINS, rot=0, trans=CONCEPT_TRANS_WEAK, scale=0)
    selector = DomainSelector(n_global=N_GLOBAL, n_source=N_SOURCE, n_target=N_TARGET,
                              n_domains_source=1, n_domains_target=1)
    return ConceptShiftDataBuilder(_init_classification, shifter, selector)


def concept_strong() -> ConceptShiftDataBuilder:
    shifter = Shifter(n_domains=CONCEPT_N_DOMAINS, rot=0, trans=CONCEPT_TRANS_STRONG, scale=0)
    selector = DomainSelector(n_global=N_GLOBAL, n_source=N_SOURCE, n_target=N_TARGET,
                              n_domains_source=1, n_domains_target=1)
    return ConceptShiftDataBuilder(_init_classification, shifter, selector)


def covariate_weak() -> CovShiftBuilder:
    selector = FeatureSelector(n_global=N_GLOBAL, n_source=N_SOURCE, n_target=N_TARGET,
                               source_scale=COV_SCALE_WEAK,
                               target_scale=COV_SCALE_WEAK,
                               bias_dist=COV_DIST_WEAK)
    return CovShiftBuilder(_init_classification, selector)


def covariate_strong() -> CovShiftBuilder:
    selector = FeatureSelector(n_global=N_GLOBAL, n_source=N_SOURCE, n_target=N_TARGET,
                               source_scale=COV_SCALE_STRONG,
                               target_scale=COV_SCALE_STRONG,
                               bias_dist=COV_DIST_STRONG)
    return CovShiftBuilder(_init_classification, selector)


bias_types = [concept_weak, concept_strong, covariate_weak, covariate_strong]
bias_names = [func.__name__ for func in bias_types]
