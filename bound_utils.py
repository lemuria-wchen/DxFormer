import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score

from data_utils import DiseaseVocab
from utils import tokenizer, save_pickle, load_pickle


def make_features(samples: list, dv: DiseaseVocab, add_imp: list, rec: float = 1.0):
    from conf import suffix
    # add_imp: a bool triple
    features, labels = [], []
    for sample in samples:
        feature = []
        for sx, attr in sample['exp_sxs'].items():
            feature.append(sx + suffix.get(attr))
        for sx, attr in sample['imp_sxs'].items():
            if np.random.rand() < rec:
                if add_imp[int(attr)]:
                    feature.append(sx + suffix.get(attr))
        features.append(' '.join(feature))
        labels.append(dv.encode(sample['label']))
    return features, labels


def run_svm_classifier(x_train, y_train, random_state: int, verbose: bool, tune: bool = True):
    if tune:
        param_grid = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]},
            {'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
        ]
        clf = GridSearchCV(SVC(random_state=random_state), param_grid, n_jobs=-1, verbose=verbose, scoring='accuracy')
        clf.fit(x_train, y_train)
    else:
        clf = SVC(random_state=random_state)
        clf.fit(x_train, y_train)
    return clf


def run_dt_classifier(x_train, y_train, random_state: int, verbose: bool, tune: bool = True):
    if tune:
        param_grid = {
            'max_depth': [2, 3, 5, 8, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'criterion': ["gini", "entropy"]
        }
        clf = GridSearchCV(
            DecisionTreeClassifier(random_state=random_state), param_grid, n_jobs=-1, verbose=verbose,
            scoring='accuracy')
        clf.fit(x_train, y_train)
    else:
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(x_train, y_train)
    return clf


def run_rf_classifier(x_train, y_train, random_state: int, verbose: bool, tune: bool = True):
    if tune:
        param_grid = {
            'max_features': ['auto', 'sqrt'],
            'max_depth': [2, 3, 5, 8, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        clf = GridSearchCV(
            RandomForestClassifier(random_state=random_state), param_grid, n_jobs=-1, verbose=verbose,
            scoring='accuracy')
        clf.fit(x_train, y_train)
    else:
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(x_train, y_train)
    return clf


def run_gbdt_classifier(x_train, y_train, random_state: int, verbose: bool, tune: bool = True):
    if tune:
        param_grid = {
            "max_features": ['sqrt'],
            "max_depth": [5, 8, 10, 12],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 3],
        }
        clf = GridSearchCV(
            GradientBoostingClassifier(random_state=random_state), param_grid, n_jobs=-1, verbose=verbose,
            scoring='accuracy')
        clf.fit(x_train, y_train)
    else:
        clf = GradientBoostingClassifier(random_state=random_state)
        clf.fit(x_train, y_train)
    return clf


def train_classifier(x_train, y_train, classifier: str, random_state: int, verbose: bool, tune: bool = True):
    classifiers = {
        'svm': run_svm_classifier,
        'dt': run_dt_classifier,
        'rf': run_rf_classifier,
        'gbdt': run_gbdt_classifier,
    }
    assert classifier in classifiers, 'The specified classifier () is not supported.'.format(classifier)
    clf = classifiers.get(classifier)(x_train, y_train, random_state, verbose, tune)
    return clf


def acc_f1_report(y_test, y_pred, digits: int):
    acc = accuracy_score(y_test, y_pred)
    ma_f1 = f1_score(y_test, y_pred, average='macro')
    wa_f1 = f1_score(y_test, y_pred, average='weighted')
    print('The accuracy/macro average f1/weighted average f1 on test set: {}/{}/{}'.format(
        round(acc, digits), round(ma_f1, digits), round(wa_f1, digits)))
    return acc, ma_f1, wa_f1


def cv_report(clf, y_test, y_pred, digits: int):
    print('Best parameters set found on development set:')
    print(clf.best_params_)
    print('Grid scores on development set:')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print('Detailed classification report:')
    print('The model is trained on the train/dev set.')
    print('The scores are computed on the test set.')
    print(classification_report(y_test, y_pred, digits=digits))


def evaluate(clf, y_test, y_pred, verbose: bool, digits: int):
    if verbose:
        cv_report(clf, y_test, y_pred, digits)
    else:
        acc_f1_report(y_test, y_pred, digits)


def build_classifier(x_train, y_train, x_test, y_test, classifier: str, random_state: int, verbose: bool, digits: int):
    # vectorizer
    cv = CountVectorizer(tokenizer=tokenizer)
    _x_train = cv.fit_transform(x_train)
    _x_test = cv.transform(x_test)
    # training
    clf = train_classifier(_x_train, y_train, classifier, random_state, verbose)
    # inference
    y_pred = clf.predict(_x_test)
    # evaluate
    evaluate(clf, y_test, y_pred, verbose, digits)
    return clf, cv


def run_classifier(train_samples: list, test_samples: list, add_imp: list, classifier: str,
                   random_state: int, verbose: bool, digits: int, path: str = None):
    # build disease vocabulary
    dv = DiseaseVocab(samples=train_samples)
    # make features
    x_train, y_train = make_features(train_samples, dv, add_imp)
    x_test, y_test = make_features(test_samples, dv, add_imp)
    clf, cv = build_classifier(x_train, y_train, x_test, y_test, classifier, random_state, verbose, digits)
    # dump classifier
    if path is not None:
        save_pickle((clf, cv), path)


def run_classifiers(train_samples: list, test_samples: list, classifier: str,
                    random_state: int, verbose: bool, digits: int):

    def _run_classifier(add_imp):
        return run_classifier(train_samples, test_samples, add_imp, classifier, random_state, verbose, digits)

    print('=' * 100 + '\n{} acc lb.\n'.format(classifier) + '=' * 100)
    _run_classifier(add_imp=[False, False, False])
    print('=' * 100 + '\n{} acc ub.\n'.format(classifier) + '=' * 100)
    _run_classifier(add_imp=[True, True, True])
    print('=' * 100 + '\n{} acc ub (pos).\n'.format(classifier) + '=' * 100)
    _run_classifier(add_imp=[False, True, False])
    print('=' * 100 + '\n{} acc ub (neg).\n'.format(classifier) + '=' * 100)
    _run_classifier(add_imp=[True, False, True])
    print('=' * 100 + '\n{} acc ub (pos + neg).\n'.format(classifier) + '=' * 100)
    _run_classifier(add_imp=[True, True, False])


def simulate(train_samples, test_samples, path, recs):
    dv = DiseaseVocab(samples=train_samples)
    clf, cv = load_pickle(path)
    acc_scores = []
    for rec in recs:
        x_test, y_test = make_features(test_samples, dv, add_imp=[True, True, True], rec=rec)
        y_pred = clf.predict(cv.transform(x_test))
        acc_scores.append(accuracy_score(y_test, y_pred))
    return acc_scores
