# -*- coding: utf-8 -*-
"""
 add lightGBM
 add CatBoost
 add other from 'classifiers_moved_from_master.py'

add INFO ABOUT DATASET like number of row, features etc
"""

"""   ABOUT HPO
If multiple fidelities are applicable:
(i.e., if it is possible to define substantially cheaper versions
of the objective function of interest, such that the performance
for these roughly correlates with the performance for the full
objective function of  interest)
    We  recommend  BOHB

If multiple fidelities are not applicable:

"""

from hyperopt import hp
import numpy as np

np.random.seed(0)


class ModelHolder:

    def __init__(self):

        self.all_models = [
            Perceptron(),
            Ridge(),
            PassiveAggressive(),
            LogisticRegression(),
            LDA(),  # + ERROR
            QDA(),  # + ERROR
            LinearSVC(),  # + часть по статье, часть auto-skl
            SVM(),  # + pm1c, по статье
            SGD(),
            KNeighbors(),  # + pm1c по статье
            NearestCentroid(),
            GaussianProcess(),
            BernoulliNB(),
            GaussianNB(),
            DecisionTree(),
            BaggingSVC(),  # + по статье hundreds of clf
            RandomForest(),  # + pm1c, по статье
            xRandTrees(),
            AdaBoost(),  # + pm1c, по статье
            HistGB(),  # +
            LabelSpreading(),  # +
            MLP(),  # +
            XGBoost(),  # + по статье
            ELM(),  # +

            DBN(),
            FactorizationMachine(),
            PolynomialNetwork()

            # DummyClassifier() #??
        ]

    # %%
    def get_approved_models(self, used_algorithms):

        approved_models = []

        used_names = []
        for name in used_algorithms:
            if used_algorithms.get(name) == True:
                used_names.append(name)

        for model in self.all_models:
            if model.short_name in used_names:
                approved_models.append(model)
                print(model.short_name)

        return approved_models

    # %%

    # estimators
    def get_all_models(self, jobs=None):
        models = []
        # just all estimators
        for m in self.all_models:
            models.append((m.short_name, m.get_skl_estimator()))

        return models


# %%
"""
#https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation
"""
from sklearn import svm


class SVM:

    def __init__(self, ):
        """
        # sci article №1

        SVM Hyperparameters range:  RBF and sigmoid (don't now about others)

        Length-scale of the kernel function, determining its locality.
          !!!gamma [2**−15,2**3](log-scale)  (~10**-4 optimal)

        Soft-margin constant, controlling the trade-off
          between model simplicity and model fit.
          !!!complexity or C [2**−5,2**15](log-scale)

        The most important hyperparameter to tune in both cases was gamma,
          followed by complexity,   but the gamma most:

        For both types of SVMs, the best performance can typically be achieved
          with low values of the gamma hyperparameter.


        # sci article №2

        In svm the biggest gain in performance can be achieved by tuning the
        kernel, gamma or degree and С

        Стратегия для HPO
        Начальный параметр C=1 шаг 10 в степени +-1, увеличивается скор,
        значит идём куда надо, находим лучшее значение. Тоже и для gamma=0.0001
        Далее фиксируем одно из максимальных значений и изменяем второе, находим
        тем самым макимум, повторяем для второго значения.
        Выбираем по итогу пару с большим скором
        """

        # for kernel=’linear’ use LinearSVC
        # split into several models
        # *Linear SVM
        # *RBF SVM
        # *Sigmoid SVM
        # *Poly SVM

        self.name = 'Support Vector Classification'
        self.short_name = 'SVM'

        self.default_parameters = {'C': 1.0, 'kernel': 'rbf', 'degree': 3,
                                   'gamma': 'scale', 'coef0': 0.0, 'shrinking': True, 'probability': False,
                                   'tol': 1e-3, 'cache_size': 200, 'class_weight': None, 'verbose': False,
                                   'max_iter': -1, 'decision_function_shape': 'ovr',
                                   'break_ties': False, 'random_state': None}

        self.parameters_range = {
            #                    'gamma':[2**-15,2**3], (log-scale)  (~10**-4 optimal)
            #                    'C'    :[2**-5,2**15], (log-scale)
            #                    'degree':[2,5]  # for ('poly')
        }

        self.scale = True

        # бывают довольно тяжелые сочетания параметров когда C очень большое
        self.search_space = {
            'name': 'SVM',
            'scale': hp.choice('SVM_scale_1', [True, False]),
            'model': svm.SVC,
            'param': hp.pchoice('SVM_kernel_type', [
                (0.65, {
                    'kernel': hp.choice('SVM_p11', ['rbf', 'sigmoid']),
                    'gamma': hp.pchoice('SVM_p12', [(0.05, 'scale'), (0.05, 'auto'),
                                                    (0.9, hp.loguniform('SVM_p121', -10.4, 2.08))]),
                    'C': hp.loguniform('SVM_p13', -3.46, 10.4),
                    'degree': 2
                }),
                (0.35, {
                    'kernel': 'poly',  # poly heavy
                    'gamma': 'scale',
                    'C': hp.loguniform('SVM_p21', -3.46, 3),  # -3.46, 5
                    'degree': hp.choice('SVM_p22', range(2, 5))
                })
            ])
        }

    def get_skl_estimator(self, **default_parameters):
        return svm.SVC(**default_parameters)


# %%

class LinearSVC:
    def __init__(self, ):
        self.description = """
                         """

        self.name = 'Linear Support Vector Classification'
        self.short_name = 'LinearSVC'

        self.default_parameters = {}
        """
        penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
        C=1.0, multi_class='ovr', fit_intercept=True,
        intercept_scaling=1, class_weight=None, verbose=0,
        random_state=None, max_iter=1000
        """

        self.search_space = {
            'name': 'LinearSVC',
            'scale': hp.choice('LinearSVC_scale_1', [True, False]),
            'model': svm.LinearSVC,
            'param': {
                'C': hp.loguniform('LinearSVC_p1', -3.46, 10.4),
                'tol': hp.loguniform('LinearSVC_p2', -11.5129, -2.30259),
                'dual': hp.choice('LinearSVC_p3', [True, False]),
                'max_iter': hp.choice('LinearSVC_p4', [1000, 2000, 4000, 8000])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return svm.LinearSVC(**default_parameters)


# %%
import xgboost


class XGBoost:
    def __init__(self, ):
        """
        generalized linear model (GLM) in xgboost - basically, using linear model, instead of tree for our boosters
        https://github.com/dmlc/xgboost/blob/master/demo/guide-python/generalized_linear_model.py

        Salient features:
            Clever penalization of trees
            A proportional shrinking of leaf nodes
            Newton Boosting
            Extra randomization parameter

        https://xgboost.readthedocs.io/en/latest/parameter.html
        kwargs ??

        # sci article №2
        For xgboost there are two parameters that are quite tunable:
            learning_rate and booster.
        """

        self.name = 'eXtreme Gradient Boosting'
        self.short_name = 'XGBoost'

        self.scale = None

        self.default_parameters = {
            "max_depth": 3, "learning_rate": 0.1, "n_estimators": 100,
            "verbosity": 1, "silent": None, "objective": "binary:logistic",
            "booster": 'gbtree', "n_jobs": 1, "nthread": None, "gamma": 0,
            "min_child_weight": 1, "max_delta_step": 0, "subsample": 1,
            "colsample_bytree": 1, "colsample_bylevel": 1, "colsample_bynode": 1,
            "reg_alpha": 0, "reg_lambda": 1, "scale_pos_weight": 1, "base_score": 0.5,
            "random_state": 0, "seed": None, "missing": None
        }
        self.parameters_mandatory_first_check = [  # from sci article №2?
            {"learning_rate": 0.018, "n_estimators": 4168},
            {"learning_rate": 0.018, "n_estimators": 4168, "subsample": 0.84,
             "max_depth": 13, "min_child_weight": 2, "colsample_bytree": 0.75,
             "colsample_bylevel": 0.58, "reg_lambda": 0.98, "reg_alpha": 1.11},
            self.default_parameters
        ]
        self.parameters_range = {
            "n_estimators": [1, 5000],
            "learning_rate": [2 ** -10, 2 ** 0],  # eta
            "subsample": [0.1, 1],
            # "booster":["gbtree","gblinear","dart"]
            "max_depth": [1, 15],
            "min_child_weight": [2 ** 0, 2 ** 7],  # 2**x
            "colsample_bytree": [0, 1],
            "colsample_bylevel": [0, 1],
            "reg_lambda": [2 ** -10, 2 ** 10],  # 2**x
            "reg_alpha": [2 ** -10, 2 ** 10],  # 2**x
        }

        # !!! split into different models by booster type?

        self.search_space = {
            'name': 'XGBoost',
            'scale': None,  # hp.choice('XGBoost_scale_1',[True,False]), #needless
            'model': xgboost.XGBClassifier,
            'param': {
                # "n_estimators":hp.randint('XGBoost_p1', 500), # was 5000
                "learning_rate": hp.loguniform('XGBoost_p2', -6.931, 0),
                # "subsample":hp.uniform('XGBoost_p3', 0.1, 1),
                # "booster": hp.choice('XGBoost_p4', ["gbtree","gblinear","dart"]),
                # 'max_depth': hp.choice('XGBoost_p5',range(1,15)),
                # "min_child_weight":hp.loguniform('XGBoost_p6', 0, 4.852),
                # "colsample_bytree":hp.uniform('XGBoost_p7', 0,1),
                # "colsample_bylevel":hp.uniform('XGBoost_p8', 0,1),
                # "reg_lambda":hp.loguniform('XGBoost_p9', -6.931, 6.931),
                # "reg_alpha":hp.loguniform('XGBoost_p10', -6.931, 6.931),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return xgboost.XGBClassifier(**default_parameters)


# %%

from sklearn import linear_model


# %%

class Perceptron:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Perceptron'
        self.short_name = 'Perceptron'

        self.scale = None

        self.default_parameters = {
            "penalty": None, "alpha": 0.0001, "fit_intercept": True,
            "max_iter": 1000, "tol": 1e-3, "shuffle": True, "verbose": 0,
            "eta0": 1.0, "n_jobs": None, "random_state": 0,
            "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "class_weight": None, "warm_start": False
        }

        self.search_space = {
            'name': 'Perceptron',
            'model': linear_model.Perceptron,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.Perceptron(**default_parameters)


# %%

class Ridge:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Ridge regression сlassifier'
        self.short_name = 'Ridge'

        self.scale = None

        self.default_parameters = {
            "alpha": 1.0, "fit_intercept": True, "normalize": False,
            "copy_X": True, "max_iter": None, "tol": 1e-3, "class_weight": None,
            "solver": "auto", "random_state": None
        }

        self.search_space = {
            'name': 'Ridge',
            'model': linear_model.RidgeClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.RidgeClassifier(**default_parameters)


# %%

class PassiveAggressive:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Passive Aggressive Classifier'
        self.short_name = 'PassiveAggressive'

        self.scale = None

        self.default_parameters = {
            "C": 1.0, "fit_intercept": True, "max_iter": 1000, "tol": 1e-3,
            "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "shuffle": True, "verbose": 0, "loss": "hinge",
            "n_jobs": None, "random_state": None, "warm_start": False,
            "class_weight": None, "average": False
        }

        self.search_space = {
            'name': 'PassiveAggressive',
            'model': linear_model.PassiveAggressiveClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.PassiveAggressiveClassifier(**default_parameters)


# %%

class LogisticRegression:

    def __init__(self, ):
        # Различная важная информация об алгоритме
        """
        LogisticRegression имеет множество солверов местами сильно влияющими на
        результат. А также Penalties that Faster for large datasets or
        Robust to unscaled datasets
        https://scikit-learn.org/stable/modules/linear_model.html
        """

        self.name = 'Logistic Regression'
        self.short_name = 'LogisticRegression'

        self.default_parameters = {'penalty': 'l2', 'dual': False, 'tol': 1e-4,
                                   'C': 1.0, 'fit_intercept': True,
                                   'intercept_scaling': 1, 'class_weight': None,
                                   'random_state': None, 'solver': 'lbfgs',
                                   'max_iter': 100, 'multi_class': 'auto',
                                   'verbose': 0, 'warm_start': False,
                                   'n_jobs': None, 'l1_ratio': None}

        self.scale = True

        self.search_space = {
            'name': 'LogisticRegression',
            'model': linear_model.LogisticRegression,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.LogisticRegression(**default_parameters)


# %%
from sklearn import discriminant_analysis


# %%

class LDA:
    def __init__(self, ):
        """
        shrinkage='auto' better i guess for all cases, because number
        of samples and dimentions almost not lower accuracy (see doc)
        """

        self.name = 'Linear Discriminant Analysis'
        self.short_name = 'LDA'

        self.scale = None

        self.default_parameters = {
            "solver": 'svd',
            "shrinkage": None,
            "priors": None,  # no point to tune in HPO
            "n_components": None,  # no point to tune in HPO
            "store_covariance": False,  # no point to tune in HPO
            "tol": 1e-4
        }

        self.search_space = {
            'name': 'LDA',
            'model': discriminant_analysis.LinearDiscriminantAnalysis,
            'param': hp.choice('LDA_solver', [
                {
                    'solver': hp.choice('LDA_p11', ['lsqr', 'eigen']),
                    'shrinkage': hp.choice('LDA_p12', [None, 'auto', hp.uniform('LDA_p121', 0, 1)]),
                    'tol': None,
                },
                {
                    'solver': 'svd',
                    'shrinkage': None,
                    'tol': hp.loguniform('LDA_p26', -10, 0),
                }
            ])
        }

    def get_skl_estimator(self, **default_parameters):
        return discriminant_analysis.LinearDiscriminantAnalysis(**default_parameters)


# %%

class QDA:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Quadratic Discriminant Analysis'
        self.short_name = 'QDA'

        self.default_parameters = {
            "priors": None,  # no point to tune in HPO
            "reg_param": 0.,
            "store_covariance": False,  # no point to tune in HPO
            "tol": 1.0e-4,  # no point to tune in HPO
        }

        self.scale = None  # ???

        self.search_space = {
            'name': 'QDA',
            'model': discriminant_analysis.QuadraticDiscriminantAnalysis,
            'param': {
                'reg_param': hp.uniform('QDA1', 0.0, 1.0),
                # 'reg_param': hp.loguniform('QDA1', -10, 0),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return discriminant_analysis.QuadraticDiscriminantAnalysis(**default_parameters)


# %%

class SGD:

    def __init__(self, ):
        """
        some examples from ???
        sgd_loss = hp.pchoice(’loss’, [(0.50, ’hinge’), (0.25, ’log’), (0.25, ’huber’)])
        sgd_penalty = hp.choice(’penalty’, [’l2’, ’elasticnet’])
        sgd_alpha = hp.loguniform(’alpha’, low=np.log(1e-5), high=np.log(1) )
        """

        self.name = 'SVM with SGD'
        self.short_name = 'SGD'

        self.scale = None

        self.default_parameters = {
            "loss": "hinge", "penalty": 'l2', "alpha": 0.0001, "l1_ratio": 0.15,
            "fit_intercept": True, "max_iter": 1000, "tol": 1e-3, "shuffle": True,
            "verbose": 0, "epsilon": 0.1, "n_jobs": None,
            "random_state": None, "learning_rate": "optimal", "eta0": 0.0,
            "power_t": 0.5, "early_stopping": False, "validation_fraction": 0.1,
            "n_iter_no_change": 5, "class_weight": None, "warm_start": False,
            "average": False
        }

        self.search_space = {
            'name': 'SGD',
            'model': linear_model.SGDClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return linear_model.SGDClassifier(**default_parameters)


# %%

from sklearn import neighbors


# %%

class KNeighbors:
    def __init__(self, n_rows=1000):
        """
        NeighborhoodComponentsAnalysis + KNeighborsClassifier  # try later
        """

        self.name = 'K-nearest neighbors classifier'
        self.short_name = 'KNeighbors'

        self.scale = None

        self.default_parameters = {
            "n_neighbors": 5, "weights": 'uniform', "algorithm": 'auto',
            "leaf_size": 30, "p": 2, "metric": 'minkowski',
            "metric_params": None, "n_jobs": None
        }

        self.parameters_mandatory_first_check = [
            {'n_neighbors': int(n_rows ** 0.5)},
            {'n_neighbors': 30},
            self.default_parameters
        ]

        self.hpo_results = []

        self.search_space = {
            'name': 'KNeighbors',
            'model': neighbors.KNeighborsClassifier,
            'param': {
                #            "n_neighbors":1+hp.randint('KNeighbors_p1', 50),
                "n_neighbors": hp.qloguniform('KNeighbors_p1', np.log(1), np.log(50), 1),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return neighbors.KNeighborsClassifier(**default_parameters)


# %%

class NearestCentroid:
    def __init__(self, ):
        """
        """

        self.name = 'Nearest centroid classifier.'
        self.short_name = 'NearestCentroid'

        self.scale = None

        self.default_parameters = {
            "metric": 'euclidean', "shrink_threshold": None
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        self.search_space = {
            'name': 'NearestCentroid',
            'model': neighbors.NearestCentroid,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return neighbors.NearestCentroid(**default_parameters)


# %%
from sklearn import gaussian_process


# %%

class GaussianProcess:
    def __init__(self, ):
        """
        from sklearn.gaussian_process.kernels import RBF  # и не только!!

        1.0 * RBF(1.0)

        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
        kernels ^
        """

        self.name = 'Gaussian Process Classifier'
        self.short_name = 'GaussianProcess'

        self.scale = None

        self.default_parameters = {
            "kernel": None, "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 0, "max_iter_predict": 100,
            "warm_start": False, "copy_X_train": True, "random_state": None,
            "multi_class": "one_vs_rest", "n_jobs": None
        }

        self.search_space = {
            'name': 'GaussianProcess',
            'model': gaussian_process.GaussianProcessClassifier,
            'param': None
        }
        # from sklearn.gaussian_process.kernels import RBF
        # GaussianProcessClassifier(1.0 * RBF(1.0)),

    def get_skl_estimator(self, **default_parameters):
        return gaussian_process.GaussianProcessClassifier(**default_parameters)


# %%

from sklearn import naive_bayes


# https://stackoverflow.com/questions/38621053/how-can-i-use-sklearn-naive-bayes-with-multiple-categorical-features

# %%

class BernoulliNB:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Naive Bayes classifier for multivariate Bernoulli models'
        self.short_name = 'BernoulliNB'

        self.scale = None

        self.default_parameters = {
            "alpha": 1.0,
            "binarize": .0,
            "fit_prior": True,
            "class_prior": None
        }

        self.search_space = {
            'name': 'BernoulliNB',
            'model': naive_bayes.BernoulliNB,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return naive_bayes.BernoulliNB(**default_parameters)


# %%

class GaussianNB:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Gaussian Naive Bayes'
        self.short_name = 'GaussianNB'

        self.scale = None

        self.default_parameters = {
            "priors": None,
            "var_smoothing": 1e-9
        }

        self.search_space = {
            'name': 'GaussianNB',
            'model': naive_bayes.GaussianNB,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return naive_bayes.GaussianNB(**default_parameters)


# %%

from sklearn import tree


# pruning (better accuracy on test set)
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

# %%

class DecisionTree:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'Decision tree classifier'
        self.short_name = 'DecisionTree'

        self.scale = None

        self.default_parameters = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": None,
            "random_state": None,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "class_weight": None,
            "presort": 'deprecated',
            "ccp_alpha": 0.0
        }

        self.search_space = {
            'name': 'DecisionTree',
            'model': tree.DecisionTreeClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return tree.DecisionTreeClassifier(**default_parameters)


# %%

from sklearn import ensemble

"""
    BrownBoost, LogitBoost

    bagging methods work best with strong and complex models
    (e.g., fully developed decision trees)

    boosting methods which usually work best with weak models
    (e.g., shallow decision trees).
"""


# %%

class BaggingSVC:  # только Bagging только SVC(kernel=rbf)
    def __init__(self, ):
        """
        base_estimator= указать нужный


        bagging methods work best with strong and complex models
        (e.g., fully developed decision trees), in contrast with
        boosting methods which usually work best with weak models
        (e.g., shallow decision trees).


        This algorithm encompasses several works from the literature:

            When random subsets of the dataset are drawn as random subsets
        of the samples, then this algorithm is known as Pasting [1].
            If samples are drawn with replacement, then the method is known
        as Bagging [2].

            When random subsets of the dataset are drawn as random subsets
        of the features, then the method is known as Random Subspaces [3].
            Finally, when base estimators are built on subsets of both
        samples and features, then the method is known as Random Patches [4].

        bootstrap = replacement
        """

        self.name = 'Bagging classifier'
        self.short_name = 'Bagging(SVС)'

        self.default_parameters = {
            "base_estimator": None,
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 1.0,
            "bootstrap": True,
            "bootstrap_features": False,
            # "oob_score":False,
            # "warm_start":False,
            # "n_jobs":None,
            # "random_state":None,
            # "verbose":0
        }

        # Gausian kernel (RBF)
        # Baggging LibSVM w
        self.search_space = {
            'name': 'Bagging(SVС)',
            'model': ensemble.BaggingClassifier,
            "scale": hp.choice('Bagging(SVС)_scale', [True, False]),
            'param': {
                "base_estimator": {
                    "model": svm.SVC,  # можно наверное усложнить и добавить больше параметров
                    "kernel": 'rbf',
                    'gamma': hp.pchoice('Bagging(SVС)_p1_gamma', [(0.05, 'scale'), (0.05, 'auto'),
                                                                  (0.9,
                                                                   hp.loguniform('Bagging(SVС)_p1_gamma_sub', -10.4,
                                                                                 2.08))]),
                    'C': hp.loguniform('Bagging(SVС)_p1_C', -3.46, 4),
                },
                "n_estimators": hp.choice('Bagging(SVС)_p2', [4, 8, 16, 32, 64]),
                # и так дефолтные нет смысла передовать
                # "max_samples":1.0,
                # "max_features":1.0,
                # "bootstrap":True,
                # "bootstrap_features":False,
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.BaggingClassifier(**default_parameters)


# %%

class RandomForest:
    def __init__(self, ):
        """
        Random forest Hyperparameters range:

        Whether to train on bootstrap samples or on the full train set.
        ??? bootstrap  {true, false}      could be useful sometimes

        Fraction of random features sampled per node
        !!! max. features  [0.1,0.9]

        The minimal number of data points required in order to create a leaf
        !!! min. samples leaf [1,20]   (1-optimal)

        The minimal number of data points required to split an internal node.
            min. samples split [2,20]

        Strategy for imputing missing numeric variables.
            imputation {mean, median, mode}

        Function to determine the quality of a possible split
            split criterion  {entropy, gini}


        the minimum samples per leaf and maximal number of features
        for determining the split were most important.

        for random forests the minimal number of data points per leaf has a
        good default and should typically be set to quite small values.

        """

        self.name = 'Random forest classifier'
        self.short_name = 'RandomForest'

        self.scale = None

        self.default_parameters = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None
        }

        self.parameters_range = {
            'max_features': [0.1, 0.9],
            'min_samples_leaf': [1, 20],  # [1-4) 1 - лучшее, дальше только хуже
            'bootstrap': [True, False]  # !!! choise not range
        }

        self.search_space = {
            'name': 'RandomForest',
            'model': ensemble.RandomForestClassifier,
            'param': {
                'max_features': hp.uniform('RandomForest_p1', 0.1, 0.9),
                'min_samples_leaf': 1 + hp.randint('RandomForest_p2', 20),
                'bootstrap': hp.choice('RandomForest_p3', [True, False])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.RandomForestClassifier(**default_parameters)


# %%

class xRandTrees:
    def __init__(self, ):
        """
        Extrimly randomised trees
        TODO check search space
        """

        self.name = 'Extra-trees classifier'
        self.short_name = 'xRandTrees'

        self.scale = None

        self.default_parameters = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.,
            "min_impurity_split": None,
            "bootstrap": False,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None
        }

        self.search_space = {
            'name': 'xRandTrees',
            'model': ensemble.ExtraTreesClassifier,
            'param': {
                'max_features': hp.uniform('xRandTrees_p1', 0.1, 0.9),
                'min_samples_leaf': 1 + hp.randint('xRandTrees_p2', 20),
                'bootstrap': hp.choice('xRandTrees_p3', [True, False])
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.ExtraTreesClassifier(**default_parameters)


# %%

class AdaBoost:
    def __init__(self, ):
        """
        maximal depth of the decision tree and, to a lesser degree, the learning rate.
        One interesting observation is that, in contrast to other
        ensembletechniques, the number of iterations did not seem to
        influenceperformance too much. The minimum value (50) appears
        to alreadybe large enough to ensure good performance, and
        increasing it doesnot lead to significantly better results.

        the maximum depth of the decision tree in Adaboost should typicallybe set to a large value
        """

        self.name = 'Adaptive Boosting classifier'
        self.short_name = 'AdaBoost'

        self.scale = None

        self.default_parameters = {
            "base_estimator": None,
            "n_estimators": 50,
            "learning_rate": 1.,
            "algorithm": 'SAMME.R',
            "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"base_estimator": DecisionTree().get_skl_estimator(max_depth=10)}
        ]

        self.parameters_range = {
            "learning_rate": [0.01, 2.0],  # (log-scale)
            "base_estimator": DecisionTree().get_skl_estimator(
                max_depth=[1, 10])  # !!! как тут быть? nested? (10 optimal maybe need more)
        }

        self.hpo_results = []

        self.search_space = {
            'name': 'AdaBoost',
            'model': ensemble.AdaBoostClassifier,
            'param': {
                "learning_rate": hp.uniform('AdaBoost_p1', 0.01, 2.0),
                "base_estimator": {  # если захочу добавить другой базовый алгоритм то hp.choise
                    "name": 'DecisionTree',
                    "model": tree.DecisionTreeClassifier,
                    "max_depth": 1 + hp.randint('AdaBoost_p2', 12)
                }
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.AdaBoostClassifier(**default_parameters)


# %%
from sklearn.experimental import enable_hist_gradient_boosting


class HistGB:
    def __init__(self, ):
        """
        # https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
        # лучше заменить на lightgbm

        This estimator is much faster than GradientBoostingClassifier
        for big datasets (n_samples >= 10 000).

        This estimator has native support for missing values (NaNs)

        etc, watch later
        """

        self.name = 'Histogram-based Gradient Boosting Classification Tree'
        self.short_name = 'HistGB'

        self.scale = None

        self.default_parameters = {
            "loss": 'auto',
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.,
            "max_bins": 255,
            "warm_start": False,
            "scoring": None,
            "validation_fraction": 0.1,
            "n_iter_no_change": None,
            "tol": 1e-7,
            "verbose": 0,
            "random_state": None
        }

        self.search_space = {
            'name': 'HistGB',
            'model': ensemble.HistGradientBoostingClassifier,
            'param': {
                "learning_rate": hp.loguniform('HistGBp1', -7, 0),
                # "max_iter":30+hp.randint('HistGBp2', 450),
                # 'max_depth': hp.choice('HistGBp3',[None,2+hp.randint('HistGBp3_1', 15) ] ),
                # 'min_samples_leaf':5+hp.randint('HistGBp4',30),
                # 'l2_regularization':hp.choice('HistGBp5',[0,0.1,0.01,0.001 ] ),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return ensemble.HistGradientBoostingClassifier(**default_parameters)


# %%
from sklearn import semi_supervised

"""
when some of the samples of training data are not labeled
I'm not sure should i use it if there are no not labeled data

Если все данные размечены то точность одинаковая
"""


# %%

class LabelSpreading:
    def __init__(self, ):
        """
        Description
        """

        self.name = 'LabelSpreading semi-supervised'
        self.short_name = 'LabelSpreading'

        self.scale = None

        self.default_parameters = {
            "kernel": 'rbf', "gamma": 20, "n_neighbors": 7, "alpha": 0.2,
            "max_iter": 30, "tol": 1e-3, "n_jobs": None
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        # TODO разделить для каждого kernel
        self.search_space = {
            'name': 'LabelSpreading',
            'model': semi_supervised.LabelSpreading,
            'param': {
                'kernel': hp.choice('LabelSpreading_p1', ['knn', 'rbf']),
                'gamma': hp.loguniform('LabelSpreading_p2', -14, 4),  # Parameter for rbf kernel. 70.4
                'n_neighbors': 1 + hp.randint('LabelSpreading_p3', 150),  # Parameter for knn kernel
                'alpha': hp.uniform('LabelSpreading_p4', 0, 1),
                'max_iter': 2 + hp.randint('LabelSpreading_p5', 150),
                'tol': hp.loguniform('LabelSpreading_p6', -10, 0),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return semi_supervised.LabelSpreading(**default_parameters)


# %%

from sklearn import neural_network


# %%

class MLP:
    def __init__(self, ):
        """
        sometimes CV train takes a long time

        # all they heavy
        models.append(('MLP', MLPClassifier()))
        models.append(('ScaledMLP', ScaledMLP))
        models.append(('PolynomialFeaturesMLP', PolynomialFeaturesMLP))
        models.append(('PolynomialFeaturesScaleMLP', PolynomialFeaturesScaleMLP))
        """

        self.name = 'Multi-layer Perceptron classifier'
        self.short_name = 'MLP'
        self.scale = None
        self.default_parameters = {
            "hidden_layer_sizes": (100,), "activation": "relu",
            "solver": 'adam', "alpha": 0.0001,
            "batch_size": 'auto', "learning_rate": "constant",
            "learning_rate_init": 0.001, "power_t": 0.5, "max_iter": 200,
            "shuffle": True, "random_state": None, "tol": 1e-4,
            "verbose": False, "warm_start": False, "momentum": 0.9,
            "nesterovs_momentum": True, "early_stopping": False,
            "validation_fraction": 0.1, "beta_1": 0.9, "beta_2": 0.999,
            "epsilon": 1e-8, "n_iter_no_change": 10, "max_fun": 15000
        }

        self.parameters_mandatory_first_check = [
            self.default_parameters
        ]

        self.search_space = {
            'name': 'MLP',
            'model': neural_network.MLPClassifier,
            'scale': hp.choice('MLP_scale_1', [True, False]),
            'param': {
                'hidden_layer_sizes': hp.choice('MLP_p1', [
                    (1 + hp.randint('size11', 300)),
                    (1 + hp.randint('size21', 200), 1 + hp.randint('size22', 200)),
                    # (1+hp.randint('size31', 200),1+hp.randint('size32', 200),1+hp.randint('size33', 200)),
                    # (1+hp.randint('size41', 200),1+hp.randint('size42', 200),1+hp.randint('size43', 200),1+hp.randint('size44', 200)),
                    # (1+hp.randint('size51', 200),1+hp.randint('size52', 200),1+hp.randint('size53', 200),1+hp.randint('size54', 200),1+hp.randint('size55', 200))
                ]),
                'activation': hp.choice('MLP_p2', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': hp.choice('MLP_p3', ['lbfgs', 'sgd', 'adam']),
                'learning_rate': hp.choice('MLP_p4', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': hp.loguniform('MLP_p5', -9, 0),
                'max_iter': 50 + hp.randint('MLP_p6', 750),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return neural_network.MLPClassifier(**default_parameters)


# %%
# import dbn

class DBN:
    def __init__(self, ):
        """
        heavy
        """

        self.name = 'Deep Belief Network Classifier'
        self.short_name = 'DBN'

        self.default_parameters = {
            "hidden_layers_structure": [100, 100],
            "activation_function": 'sigmoid',
            "optimization_algorithm": 'sgd',
            "learning_rate": 1e-3,
            "learning_rate_rbm": 1e-3,
            "n_iter_backprop": 100,
            "l2_regularization": 1.0,
            "n_epochs_rbm": 10,
            "contrastive_divergence_iter": 1,
            "batch_size": 32,
            "dropout_p": 0,  # float between 0 and 1.
            "verbose": False
        }

        self.search_space = {
            'name': 'DBN',
            'model': None,  # dbn.SupervisedDBNClassification,
            'param': None,
        }

    # def get_skl_estimator(self, **default_parameters):
    #    return dbn.SupervisedDBNClassification(**default_parameters)


# %%

import polylearn


# %%

class FactorizationMachine:
    def __init__(self, ):
        """
        http://mblondel.org/publications/mblondel-icml2016.pdf

        """

        self.name = 'Factorization Machine Classifier'
        self.short_name = 'FactorizationMachine'

        self.default_parameters = {
            "degree": 2,
            "loss": 'squared_hinge',
            "n_components": 2,
            "alpha": 1,
            "beta": 1,
            "tol": 1e-6,
            "fit_lower": 'explicit',
            "fit_linear": True,
            "warm_start": False,
            "init_lambdas": 'ones',
            "max_iter": 10000,
            "verbose": False,
            "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"n_components": 1},
            {"n_components": 2},
            {"n_components": 3}
        ]

        self.search_space = {
            'name': 'FactorizationMachine',
            'model': polylearn.FactorizationMachineClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return polylearn.FactorizationMachineClassifier(**default_parameters)


# %%

class PolynomialNetwork:
    def __init__(self, ):
        """
        http://mblondel.org/publications/mblondel-icml2016.pdf

        """

        self.name = 'Polynomial Network Classifier'
        self.short_name = 'PolynomialNetwork'

        self.default_parameters = {
            "degree": 2, "loss": 'squared_hinge', "n_components": 2, "beta": 1,
            "tol": 1e-6, "fit_lower": 'augment', "warm_start": False,
            "max_iter": 10000, "verbose": False, "random_state": None
        }

        self.parameters_mandatory_first_check = [
            {"degree": 2},
            {"degree": 3}
        ]

        self.search_space = {
            'name': 'PolynomialNetwork',
            'model': polylearn.PolynomialNetworkClassifier,
            'param': None
        }

    def get_skl_estimator(self, **default_parameters):
        return polylearn.PolynomialNetworkClassifier(**default_parameters)


# %%

import elm


class ELM:
    def __init__(self, ):
        """
        http://mblondel.org/publications/mblondel-icml2016.pdf

        very fast and good accuracy
        may be try with AdaBoost or Boosting?
        """

        self.name = 'Extreme learning machine'
        self.short_name = 'ELM'

        self.default_parameters = {
            "hid_num": 10,
            "a": 1
        }

        self.scale = None
        self.search_space = {
            'name': 'ELM',
            'model': elm.ELM,
            'param': {
                'hid_num': hp.qloguniform('ELM_p1', np.log(1), np.log(1500), 1),
                'a': hp.loguniform('ELM_p2', -12, 6),
            }
        }

    def get_skl_estimator(self, **default_parameters):
        return elm.ELM(hid_num=10)  # (**default_parameters)


# %%                    ???

"""
from sklearn import dummy

class DummyClassifier:
    def __init__(self, ):

        self.name = 'Dummy Classifier'
        self.short_name = 'Dummy'

        self.scale = None

        self.default_parameters={
            "strategy":"warn",
            "random_state":None,
            "constant":None
                }

        self.parameters_mandatory_first_check=[
                {'strategy':'stratified'},
                {'strategy':'most_frequent'},
                {'strategy':'prior'},
                {'strategy':'uniform'}
                ]


    def get_skl_estimator(self, **default_parameters):
        return dummy.DummyClassifier(**default_parameters)

"""
