# this script is to train a machine learning based disease classifier.
# it is used to calculate the boundary of disease diagnosis accuracy.

# symptom set: {symptom_1, symptom_2, ..., symptom_n}
# attr set: {yes, no, not_sure}
# disease set: {disease_1, ..., disease_m}
# explicit symptoms: {symptom_1: attr_1, ..., symptom_k: attr_k}
# implicit symptoms: {symptom_k+1: attr_k+1, ..., symptom_n: attr_n}
# lower bound: train a disease classifier only use explicit symptoms
# upper bound: train a disease classifier use explicit plus implicit symptoms

from utils import load_data, make_dirs
from bound_utils import run_classifier, run_classifiers
from conf import digits


# parameters for calculating empirical upper and lower bounds
ds = 'mz4'
classifier = 'svm'
random_state = 123
verbose = False


# load dataset
train_path, test_path = 'data/{}/train_set.json'.format(ds), 'data/{}/test_set.json'.format(ds)
train_samples, test_samples = load_data(train_path), load_data(test_path)

# ablation experiments of classifiers
run_classifiers(train_samples, test_samples, classifier, random_state, verbose, digits)


# train ub-classifier (with complete implicit symptoms)
# path = 'saved/{}/{}.pkl'.format(ds, classifier)
# make_dirs(path)
# print('=' * 100 + '\n{} acc ub.\n'.format(classifier) + '=' * 100)
# run_classifier(train_samples, test_samples, add_imp=[True, True, True], classifier=classifier,
#                random_state=random_state, verbose=verbose, digits=digits, path=path)

# results
#           acc-lb  acc-ub  acc-pos acc-neg acc-pos+neg
# dxy       0.644   0.856   0.808   0.663   0.856
# mz4       0.648   0.697   0.704   0.676   0.697
# mz4-1k    0.646   0.757   0.693   0.693   0.757
# mz10      0.490   0.671   0.592   0.578   0.603
# mz10-old  0.500   0.706   0.619   0.536   0.641
