import os

from datagen.covshift.builder import CovShiftBuilder
from datagen.covshift.selector import FeatureSelector
from datagen.conceptshift.builder import ConceptShiftDataBuilder
from datagen.conceptshift.selector import DomainSelector
from datagen.conceptshift.shifter import Shifter
from util.batch import batch_generate

n_samples = 2000
n_global = 1000
n_source = 100
n_target = 500

n_features = 3
n_informative=3

CONCEPT_N_DOMAINS = 2
CONCEPT_TRANS_WEAK = 2
CONCEPT_TRANS_STRONG = 4
CONCEPT_TRANS = 4

COV_SCALE_STRONG = 0.75
COV_DIST_STRONG = 1
COV_SCALE = 0.75
COV_DIST = 1

concept=True

if concept:
    batch_path = os.path.join(os.getcwd(), 'data', 'emiel', f'emiel_{n_samples}|{n_features}|{n_informative}', 
                            f'emiel_{CONCEPT_N_DOMAINS}|{CONCEPT_TRANS}_{n_global}|{n_source}|{n_target}')

    init_classification = dict(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                            n_repeated=0, n_redundant=n_features-n_informative, n_clusters_per_class=4)

    shifter = Shifter(n_domains=CONCEPT_N_DOMAINS, rot=0, trans=CONCEPT_TRANS, scale=0)
    selector = DomainSelector(n_global=n_global, n_source=n_source, n_target=n_target,
                                n_domains_source=1, n_domains_target=1)
    builder = ConceptShiftDataBuilder(init_classification, shifter, selector)
    batch_generate(builder, 10, batch_path)
else:
    batch_path = os.path.join(os.getcwd(), 'data', 'emiel', f'emiel_{}|{n_samples}|{n_features}|{n_informative}', 
                            f'emiel_cov_{COV_SCALE}|{COV_DIST}_{n_global}|{n_source}|{n_target}')
    init_classification = dict(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                            n_repeated=0, n_redundant=n_features-n_informative, n_clusters_per_class=4)
    selector = FeatureSelector(n_global=n_global, n_source=n_source, n_target=n_target,
                                source_scale=COV_SCALE,
                                target_scale=COV_SCALE,
                                bias_dist=COV_DIST)
    return CovShiftBuilder(init_classification, selector)
