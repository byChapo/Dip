# -*- coding: utf-8 -*-

import warnings
import numbers
import time
from traceback import format_exception_only
from contextlib import suppress

import numpy as np
from joblib import Parallel, delayed

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable,_message_with_time
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
#from sklearn.utils.validation import _is_arraylike

import pickle
import sys
import pandas as pd




__all__ = ['cross_validate', 'cross_val_score', 'split_val_score']


#!!! IN CASE OF linear_model.LogisticRegressionCV YOU SHOULD MODIFY CV FUNCTION add parametr embeded CV or something else


def split_val_score(estimator, X_train, X_test, Y_train, Y_test, scoring):
    full_time_start = time.perf_counter()

    scorer = check_scoring(estimator, scoring=scoring)

    results={}

    fit_time_start = time.perf_counter()
    estimator.fit(X_train,Y_train)
    results['fit_time']=time.perf_counter()-fit_time_start


    p = pickle.dumps(estimator)
    results['memory_fited']= sys.getsizeof(p) #in bytes



    score_time_start=time.perf_counter()
#    Y_pred=estimator.predict(X_test)
#    if(metrics=='accuracy'): #??? протестировать
#        from sklearn.metrics import accuracy_score
#        results['test_score']=accuracy_score(Y_test, Y_pred)
    results['test_score'] = _score(estimator, X_test, Y_test, scorer)
    results['score_time'] = time.perf_counter()-score_time_start



    #TODO переделать без костылей
    #for numpy
    if(isinstance(X_test, np.ndarray)):
        inf_example=X_test[0].reshape(1, -1)

    #for DataFrame
    if(isinstance(X_test, pd.DataFrame)):
        inf_example=X_test.iloc[0].to_numpy().reshape(1, -1)




    inference_time_start=time.perf_counter()
    s=estimator.predict(inf_example)
    results['inference_time']=time.perf_counter()-inference_time_start


    results['full_time']= time.perf_counter()-full_time_start
    return results





def cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None,
                   n_jobs=None, verbose=0, fit_params=None,
                   pre_dispatch='2*n_jobs', return_train_score=False,
                   return_estimator=False, error_score=np.nan):

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times, inference_times, memory, total_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    ret = {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)
    ret['inference_time'] = np.array(inference_times)
    ret['memory_fited'] = np.array(memory)
    ret['total_time'] = np.array(total_times)


    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])

    return ret



def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=None, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', error_score=np.nan):

    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                scoring={'score': scorer}, cv=cv,
                                n_jobs=n_jobs, verbose=verbose,
                                fit_params=fit_params,
                                pre_dispatch=pre_dispatch,
                                error_score=error_score)


    return cv_results





def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=True, return_estimator=False,
                   error_score=np.nan):

    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    train_scores = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.perf_counter()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)



    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)


    except Exception as e:
        # Note fit time as time until error
        fit_time = time.perf_counter() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exception_only(type(e), e)[0]),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.perf_counter() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer)




        score_time = time.perf_counter() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer)
    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))

    total_time = score_time + fit_time
    if verbose > 1:
        print(_message_with_time('CV', msg, total_time))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]


    p = pickle.dumps(estimator)
    mem=sys.getsizeof(p) #in bytes


    #TODO переделать без костылей
    #for numpy
    if(isinstance(X_test, np.ndarray)):
        inf_example=X_test[0].reshape(1, -1)

    #for DataFrame
    if(isinstance(X_test, pd.DataFrame)):
        inf_example=X_test.iloc[0].to_numpy().reshape(1, -1)
    #Reshape your data either using array.reshape(-1, 1)
    #if your data has a single feature or array.reshape(1, -1) if it contains
    #a single sample.


    inference_time_start=time.perf_counter()
    s=estimator.predict(inf_example)
    inference_time_end=time.perf_counter()
    inference_time=inference_time_end-inference_time_start


    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time,inference_time, mem, total_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret








def _score(estimator, X_test, y_test, scorer):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)
    if y_test is None:
        scores = scorer(estimator, X_test)
    else:
        scores = scorer(estimator, X_test, y_test)

    error_msg = ("scoring must return a number, got %s (%s) "
                 "instead. (scorer=%s)")
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, 'item'):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, 'item'):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores












def _aggregate_score_dicts(scores):
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}
