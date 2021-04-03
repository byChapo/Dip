# -*- coding: utf-8 -*-

from time import perf_counter
from models import ModelHolder
import numpy as np
import pandas as pd
import os


class ModelSelection:
    def __init__(self, experiment_name, duration, min_accuracy,
                 max_model_memory, max_prediction_time, max_train_time,
                 used_algorithms, metric, validation, iterations,
                 initial_resampling=None, max_jobs=1):
        print('!start!')

        # !!!  DEV
        # TODO initial_resampling:
        #  None
        #  'under'    - Under-sampling
        #  'over'     - Over-sampling
        #  'combined' - Over-sampling followed by under-sampling
        #  'all'      - try all options (3)
        #  GOAL: find single the best resampling and use as initial self.x, self.y
        self.initial_resampling = initial_resampling  # combine with balanced_accuracy metric ??

        self.row_count = None
        self.columns_count = None  # all col (with target?)
        self.target_column = None
        self.cat_columns = []
        self.path_to_save = None

        self.max_jobs = max_jobs
        self.CV_jobs = self.max_jobs # fast solution TODO make better, kinda resource manager
        # !!!  DEV


        self.experiment_name = experiment_name
        self.duration = duration
        self.min_accuracy = min_accuracy
        self.max_model_memory = max_model_memory
        self.max_prediction_time = max_prediction_time
        self.max_train_time = max_train_time
        self.iterations = iterations

        self.used_algorithms = used_algorithms
        self.validation = validation

        # tested: accuracy, roc_auc, balanced_accuracy
        # but currently some problems with f1, recall, precision
        self.metric = metric

        self.time_end = perf_counter() + duration

        self.valtype = ''
        self.cv_splits = None

        # TODO change
        if self.validation in ["3 fold CV", "5 fold CV", "10 fold CV"]:
            if self.validation == "3 fold CV":
                self.cv_splits = 3
            elif self.validation == "5 fold CV":
                self.cv_splits = 5
            elif self.validation == "10 fold CV":
                self.cv_splits = 10
            self.valtype = 'CV'
            from sklearn import model_selection
            self.kfold = model_selection.KFold(n_splits=self.cv_splits)
        elif self.validation == "holdout":
            self.valtype = 'H'


        self.models = ModelHolder().get_approved_models(self.used_algorithms)

        print('!end!')



    def check_time(self):
        if self.time_end > perf_counter():
            return True
        else:
            return False



    def fit(self, x, y, num_features=[], cat_features=[], txt_features=[]):
        """
        x may include 'y' and any other even unused columns
        """
        from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, STATUS_FAIL
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from utility.util import split_val_score, cross_val_score



        # TODO DEV
        from data_preprocessing import DataPreprocessing
        preproc = DataPreprocessing(x, num_features, cat_features)
        self.x = preproc.get_x()
        # self.nrows, self.ncol = self.x.shape # TODO ?
        self.y = y
        print('class DataPreprocessing: Done')


        from resampling import initial_resample
        if self.initial_resampling != None:
            print('RESAMPLING start')
            self.x,self.y = initial_resample(self.x.copy(),self.y.copy(),'all')
            print('RESAMPLING end')


        from data_preprocessing import encode_y_ELM_binary
        if self.used_algorithms['ELM'] == True:
            self.y_ELM = encode_y_ELM_binary(self.y)
            self.x_ELM = self.x.copy()
            self.x_ELM = self.x_ELM.astype(np.float64)
        # TODO DEV



        # TODO change
        # if validation == holdout
        if self.valtype == 'H':
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self.x, self.y, test_size=0.2)

            if self.used_algorithms['ELM'] == True:
                self.x_train_ELM, self.x_test_ELM, self.y_train_ELM, \
                self.y_test_ELM = train_test_split(self.x.astype(np.float64),
                                                   self.y_ELM, test_size=0.2)

        # %%
        def objective_func(args):
            if self.check_time() == True:

                # debug
                print(args['name'], args['param'])

                # every commented parametr worsen performans on G-credit
                # better without them
                if args['name'] == 'SVM':
                    clf = args['model'](
                        kernel=args['param']['kernel'],
                        gamma=args['param']['gamma'],
                        C=args['param']['C'],
                        degree=args['param']['degree']
                    )
                    if args['scale'] == True:
                        clf = make_pipeline(StandardScaler(), clf)

                elif args['name'] == 'XGBoost':
                    clf = args['model'](
                        learning_rate=args['param']['learning_rate'],
                        # low efficiency
                        # booster = args['param']['booster'],
                        # n_estimators = args['param']['n_estimators'],
                        # subsample = args['param']['subsample'],
                        # max_depth = args['param']['max_depth'],
                        # min_child_weight = args['param']['min_child_weight'],
                        # colsample_bytree = args['param']['colsample_bytree'],
                        # colsample_bylevel = args['param']['colsample_bylevel'],
                        # reg_lambda = args['param']['reg_lambda']  ,
                        # reg_alpha = args['param']['reg_alpha']  ,
                    )
                    # scale removed

                elif args['name'] == 'RandomForest':
                    clf = args['model'](
                        max_features=args['param']['max_features'],
                        min_samples_leaf=args['param']['min_samples_leaf'],
                        bootstrap=args['param']['bootstrap'])

                elif args['name'] == 'KNeighbors':
                    clf = args['model'](
                        n_neighbors=int(args['param']['n_neighbors'])
                    )

                elif args['name'] == 'AdaBoost':
                    if args['param']['base_estimator']['name'] == 'DecisionTree':
                        base = args['param']['base_estimator']['model'](
                            max_depth=args['param']['base_estimator']['max_depth'])
                    clf = args['model'](
                        learning_rate=args['param']['learning_rate'],
                        base_estimator=base)

                elif args['name'] == 'LinearSVC':
                    clf = args['model'](
                        C=args['param']['C'],
                        tol=args['param']['tol'],
                        dual=args['param']['dual'],
                        max_iter=args['param']['max_iter'])
                    if args['scale'] == True:
                        clf = make_pipeline(StandardScaler(), clf)

                elif args['name'] == 'HistGB':
                    clf = args['model'](
                        learning_rate=args['param']['learning_rate'],
                        # max_iter =  args['param']['max_iter'],
                        # max_depth = args['param']['max_depth'],
                        # min_samples_leaf = args['param']['min_samples_leaf'],
                        # l2_regularization = args['param']['l2_regularization'],
                    )

                elif args['name'] == 'MLP':
                    clf = args['model'](
                        hidden_layer_sizes=args['param']['hidden_layer_sizes'],
                        activation=args['param']['activation'],
                        solver=args['param']['solver'],
                        learning_rate=args['param']['learning_rate'],
                        learning_rate_init=args['param']['learning_rate_init'],
                        max_iter=args['param']['max_iter'],
                    )

                    if args['scale'] == True:
                        clf = make_pipeline(StandardScaler(), clf)

                elif args['name'] == 'LabelSpreading':
                    clf = args['model'](
                        kernel=args['param']['kernel'],
                        gamma=args['param']['gamma'],
                        n_neighbors=args['param']['n_neighbors'],
                        alpha=args['param']['alpha'],
                        max_iter=args['param']['max_iter'],
                        tol=args['param']['tol'],
                    )

                elif args['name'] == 'LDA':
                    clf = args['model'](
                        solver=args['param']['solver'],
                        shrinkage=args['param']['shrinkage'],
                        tol=args['param']['tol'],
                        # priors, n_components, store_covariance не нужены
                    )

                elif args['name'] == 'QDA':
                    clf = args['model'](
                        reg_param=args['param']['reg_param'],
                    )

                elif args['name'] == 'ELM':
                    # TODO -1 1
                    clf = args['model'](
                        hid_num=int(args['param']['hid_num']),
                        a=args['param']['a'],
                    )

                elif args['name'] == 'Bagging(SVC)':  # rbf
                    base = args['param']['base_estimator']['model'](
                        kernel=args['param']['base_estimator']['kernel'],
                        gamma=args['param']['base_estimator']['gamma'],
                        C=args['param']['base_estimator']['C'],
                    )
                    clf = args['model'](
                        base_estimator=base,
                        n_estimators=args['param']['n_estimators'],
                    )
                    if args['scale'] == True:
                        clf = make_pipeline(StandardScaler(), clf)

                elif args['name'] == 'xRandTrees':
                    clf = args['model'](
                        max_features=args['param']['max_features'],
                        min_samples_leaf=args['param']['min_samples_leaf'],
                        bootstrap=args['param']['bootstrap'],
                        # TODO add more? check existing
                    )

                else:
                    clf = args['model']()
                    # TODO add other

                # %%
                if self.valtype == 'CV':
                    start_timer = perf_counter()

                    if args['name'] == 'ELM':
                        # if ValueError
                        try:
                            cv_results = cross_val_score(clf, self.x_ELM, self.y_ELM, cv=self.kfold,
                                                         scoring=self.metric, n_jobs=self.CV_jobs)
                        except:  # ValueError
                            print("Oops! Error...")
                            cv_results = {}
                            cv_results['memory_fited'] = np.array([9999999999, 9999999999])
                            cv_results['inference_time'] = np.array([9999999999, 9999999999])
                            cv_results['test_score'] = np.array([-9999999999, -9999999999])
                    else:
                        cv_results = cross_val_score(clf, self.x, self.y, cv=self.kfold, scoring=self.metric,
                                                     n_jobs=self.CV_jobs)

                    mem = cv_results['memory_fited'].max()
                    pred_time = cv_results['inference_time'].max()
                    accuracy = cv_results['test_score'].mean()
                    time_all = perf_counter() - start_timer
                # %%
                elif self.valtype == 'H':
                    start_timer = perf_counter()

                    if args['name'] == 'ELM':
                        # TODO ValueError
                        try:
                            results = split_val_score(clf, self.x_train_ELM, self.x_test_ELM, self.y_train_ELM,
                                                      self.y_test_ELM, scoring=self.metric)
                        except:  # ValueError
                            print("Oops! Error...")
                            results = {}
                            results['memory_fited'] = 9999999999
                            results['inference_time'] = 9999999999
                            results['test_score'] = -9999999999
                    else:
                        results = split_val_score(clf, self.x_train, self.x_test, self.y_train, self.y_test,
                                                  scoring=self.metric)

                    pred_time = results['inference_time']
                    mem = results['memory_fited']
                    accuracy = results['test_score']
                    time_all = perf_counter() - start_timer
                # %%
                loss = (-accuracy)

                # monitoring
                print(accuracy)
                print('')

                # Model requirements check
                if (accuracy < self.min_accuracy or
                        mem > self.max_model_memory or
                        pred_time > self.max_prediction_time or
                        time_all > self.max_train_time):
                    status = STATUS_FAIL
                    loss = 999
                else:
                    status = STATUS_OK

                return {
                    'loss': loss,
                    'status': status,
                    'accuracy': accuracy,
                    'model_memory': mem,
                    'prediction_time': pred_time,
                    'train_time': time_all,
                    'model_name': args['name'],
                    'model': clf
                }
            else:
                return {
                    'loss': None,
                    'status': STATUS_FAIL,
                    'accuracy': None,
                    'model_memory': None,
                    'prediction_time': None,
                    'train_time': None,
                    'model_name': None,
                    'model': None
                }

        # %%

        # Prepairing to search
        trials = Trials()
        hyper_space_list = []
        for model in self.models:
            hyper_space_list.append(model.search_space)

        space = hp.choice('classifier', hyper_space_list)

        # Start search
        import hyperopt

        try:
            fmin(objective_func, space, algo=tpe.suggest, max_evals=self.iterations, trials=trials)
            self.status = 'OK'
        except hyperopt.exceptions.AllTrialsFailed:
            print('No solutions found. Try a different algorithm or change the requirements')
            self.status = 'No solutions found'
        # except:
        #    print('Unexpected error')
        #    self.status='Unexpected error'

        # %%
        if self.status == 'OK': # TODO remove this filter?
            # SAVE to EXCEL
            excel_results = []
            for res in trials.results:
                excel_results.append((res['accuracy'], res['model'], res['model_name'], res['model_memory'],
                                      res['prediction_time'], res['train_time']))

            self.results_excel = pd.DataFrame(excel_results,
                                              columns=['accuracy', 'model', 'model_name', 'model_memory',
                                                       'prediction_time', 'train_time'])

            # save results with only ok status
            results = []
            for res in trials.results:
                if res['status'] == 'ok':
                    results.append((res['accuracy'], res['model'], res['model_name'], res['model_memory'],
                                    res['prediction_time'], res['train_time']))

            self.optimal_results = results



    def save_results(self, n_best='All', save_excel=True, save_config=True):

        def save_model(to_persist, name):
            dir_name = self.experiment_name
            work_path = os.getcwd()
            path = os.path.join(work_path, dir_name)
            print('Save model: ' + name)
            if os.path.exists(path) == False:
                os.mkdir(path)
            savedir = path
            filename = os.path.join(savedir, name + '.joblib')
            import joblib
            joblib.dump(to_persist, filename)

        # func for sort self.optimal_results
        def sortSecond(val):
            return val[0]


        # Create folder
        work_path = os.getcwd()
        path = os.path.join(work_path, self.experiment_name)
        if os.path.exists(path) == False:
            os.mkdir(path)


        # sort self.optimal_results by accuracy
        self.optimal_results.sort(key=sortSecond, reverse=True)


        # TODO probably need rework. Looks not optimal.
        if n_best == "All":
            for i in range(len(self.optimal_results)):
                model = self.optimal_results[i][1]
                name = str(i + 1) + '_' + str(self.optimal_results[i][2]) + '_' + str(self.optimal_results[i][0])
                save_model(model, name)
        else:
            if isinstance(n_best, int):
                model_num = n_best
            elif n_best == None: # TODO NEW, need test
                model_num = None
            elif n_best == "The best":
                model_num = 1
            elif n_best == "Top 5":
                model_num = 5
            elif n_best == "Top 10":
                model_num = 10
            elif n_best == "Top 25":
                model_num = 25
            elif n_best == "Top 50":
                model_num = 50

            if model_num != None:
                if len(self.optimal_results) < model_num:
                    model_num = len(self.optimal_results)

                for i in range(model_num):
                    model = self.optimal_results[i][1]
                    name = str(i + 1) + '_' + str(self.optimal_results[i][2]) + '_' + str(self.optimal_results[i][0])
                    save_model(model, name)

        if save_excel == True:
            self.results_excel.sort_values(by='accuracy', ascending=False, inplace=True)
            self.results_excel.to_excel(self.experiment_name + "\\" + self.experiment_name + "_results.xlsx")

        if save_config == True:
            import config
            cfg = config.default_config.copy()

            # need because when use api you don't have default config.json
            cfg['task'] = 'classification'
            cfg['experiment_name'] = self.experiment_name
            cfg['model_requirements']['min_accuracy'] = self.min_accuracy
            cfg['model_requirements']['max_memory'] = self.max_model_memory
            cfg['model_requirements']['max_single_predict_time'] = self.max_prediction_time
            cfg['model_requirements']['max_train_time'] = self.max_train_time
            cfg['search_space'] = self.used_algorithms
            cfg['search_options']['duration'] = self.duration
            cfg['search_options']['iterations'] = self.iterations
            cfg['search_options']['metric'] = self.metric
            cfg['search_options']['validation'] = self.validation
            cfg['search_options']['saved_top_models_amount'] = n_best
            cfg['paths']['DS_abs_path'] = None
            cfg['paths']['CD_abs_path'] = None

            config.save_config(cfg, self.experiment_name + '\\config.json')


# ['accuracy']['model']['model_name']['model_memory']['prediction_time']['train_time']


