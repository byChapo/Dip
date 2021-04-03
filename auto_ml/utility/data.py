# -*- coding: utf-8 -*-
"""
Module is used for data loading
    * your datasets
    * public datasets from OpenML, UCI, (Scikit-Learn)

dataset (ds) - file (table) containing X, y
column description (cd) -
"""

import numpy as np
import pandas as pd
import os

np.random.seed(0)


def load_DS_as_df(path):  # Load DataSet from disc as pandas DataFrame
    # path = str(Path(path))
    if os.path.isfile(path):
        DS = pd.read_csv(path, skiprows=0)  # .dropna(how='any').as_matrix() #TODO change .as_matrix() to .values
        return DS


def load_DS_as_np(path):  # TODO ? or not
    # TODO? self.DS = np.genfromtxt(path, delimiter=',')   # ,dtype=None, encoding=None
    if os.path.isfile(path):
        DS = pd.read_csv(path, skiprows=0)
        return DS.values


#########################################

def load_CD_as_list(path):  # Load Column Description from disc as list
    # path = str(Path(path))
    if os.path.isfile(path):
        numpy_cd = np.genfromtxt(path, delimiter=',', dtype=None, encoding=None)
        CD = numpy_cd.tolist()
        return CD


def CD_processing(path):
    CD = load_CD_as_list(path)

    num_cols = []
    cat_cols = []
    txt_cols = []
    label_col = None

    for column in CD:
        if column[1] == 'Num':
            num_cols.append(column[0] - 1)
        elif column[1] == 'Categ':
            cat_cols.append(column[0] - 1)
        elif column[1] == 'Text':
            txt_cols.append(column[0] - 1)
        elif column[1] == 'Label':  # TODO change to cols?
            label_col = column[0] - 1

    return num_cols, cat_cols, txt_cols, label_col


#########################################


def load_dataset_pandas(ds_abs_path, ds_type):
    # https://www.kdnuggets.com/2019/11/speed-up-pandas-4x.html
    # !!! pip install modin[dask]
    #  modin нужен так как: работает как pandas но быстрее, DataFrame designed for
    #  datasets from 1MB to 1TB+

    # dask DataFrame нужен если датасет не влезает в память и?
    # https://docs.dask.org/en/latest/dataframe.html
    """
    pd.read_clipboard pd.read_excel pd.read_feather
    pd.read_fwf pd.read_gbq pd.read_hdf pd.read_html pd.read_json
    pd.read_msgpack pd.read_parquet pd.read_pickle pd.read_sas
    pd.read_spss pd.read_sql pd.read_sql_query pd.read_sql_table
    pd.read_stata pd.read_table

    В будущем можно расширить
    """

    if ds_type == '.csv':
        # pandas_ds = pd.read_csv(ds_abs_path)
        pandas_ds = 0
        # final type should be ndarray?
        # pandas DataFrame to Numpy
        # np.asarray(df)

        return pandas_ds


#    df['origin'] = df['origin'].astype('category')
# in Python, it's a good practice to typecast categorical features to a
# category dtype because they make the operations on such columns much
# faster than the object dtype
#    return 'Функция ещё не дописана'

# %%   load with numpy

def load_dataset_numpy(ds_abs_path, ds_type):
    if ds_type == '.csv':
        from numpy import genfromtxt
        numpy_ds = genfromtxt(ds_abs_path, delimiter=',')
        return numpy_ds
    else:
        return 'this type is not supported'


# %%

def load_CD(cd_abs_path, cd_type):
    if cd_type == '.csv':
        from numpy import genfromtxt
        numpy_cd = genfromtxt(cd_abs_path, delimiter=',', dtype=None, encoding=None)
        cd = numpy_cd.tolist()
    else:
        return 'this type is not supported'

    return cd


# %%   если в CD указаны не все колонки то недостоющие будут дополнены как Auxiliary

def complete_CD(cd, ds, lib):
    if lib == 'numpy':
        rows_count = ds.shape[0]
        columns_count = ds.shape[1]
        pass


    elif lib == 'pandas':
        print('pandas not supported')

    completed_cd = 0
    return completed_cd


# %%

def load_X_y_CD(ds_abs_path, ds_type, cd_abs_path, cd_type, lib='numpy'):
    if lib == 'numpy':
        ds = load_dataset_numpy(ds_abs_path, ds_type)
        cd = complete_CD(load_CD(cd_abs_path, cd_type), ds, lib)

        X = 1
        y = 1
        cd = 1

        return X, y, cd

    elif lib == 'pandas':
        print('pandas not supported')


def load_openml_dataset(data_set_type='binary'):
    # add OpenML100 or OpenML-CC18
    # pip install openml
    # https://docs.openml.org/benchmark/
    """
    ############################
    #
    #        НЕ РАБОТАЕТ проблема в pandas?
    #
    ###########################№
    """
    # https://scikit-learn.org/stable/datasets/index.html#openml
    from sklearn.datasets import fetch_openml
    X, Y = fetch_openml(name='cylinder-bands', return_X_y=True, as_frame=True)

    print(X.shape, Y.shape)

    # TODO REMOVE and use imputer as a part of pipeline
    from missing import delete_missing
    X, Y = delete_missing(X, Y)
    print(X.shape, Y.shape)
    #        X.to_excel("X.xlsx")
    #        Y.to_excel("Y.xlsx")

    X_cat = X.select_dtypes(include=['category'])
    X_other = X.select_dtypes(exclude=['category'])

    from encoding import unsupervised_encoding

    X_unsu_encoded = unsupervised_encoding(X_cat)

    print(X.shape, Y.shape)
    print(X, Y)

    X_concat = pd.concat([X_other, X_unsu_encoded], axis=1, sort=False)

    X = X_concat.to_numpy()
    from encoding import encode_y
    Y = encode_y(Y)

    """
    data = fetch_openml(name='cylinder-bands')
    print(data.data.shape,"\n")
    print(data.target.shape,"\n")
    print(np.unique(data.target),"\n")
    print(data.details)
    """
    return X, Y


# %%

def load_example_dataset(data_set_type='binary'):
    if data_set_type == 'binary':

        from sklearn.datasets import load_breast_cancer
        X, Y = load_breast_cancer(return_X_y=True)

        #        from sklearn.datasets import make_hastie_10_2
        #        X, Y = make_hastie_10_2(n_samples=12000, random_state=1)

        #        from sklearn.datasets import make_moons
        #        X,Y=make_moons(noise=0.3, random_state=0)

        #        from sklearn.datasets import make_circles
        #        X,Y=make_circles(noise=0.2, factor=0.5, random_state=1)

        #        from sklearn.datasets import make_classification
        #        noise_col=0
        #        informative_col=10
        #        #false_y#flip_y
        #        #weights=None, flip_y=0.01, class_sep=1.0, hypercube=True,
        #        #shift=0.0, scale=1.0, shuffle=True
        #        X, Y = make_classification(n_repeated=0,n_classes=2,
        #               n_samples=1000,n_features=noise_col+informative_col,
        #               n_redundant=noise_col, n_informative=informative_col,
        #               random_state=1, n_clusters_per_class=2)

        #        df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
        #        array = df.values
        #        X = array[:,0:8]
        #        Y = array[:,8]

        #        genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
        #                                   'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
        #                                   sep='\t', compression='gzip')
        #        X, Y = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

        #        from sklearn.datasets import make_blobs
        #        X, Y = make_blobs(n_samples=120,n_features=5, centers=2)

        #        from sklearn.datasets import make_gaussian_quantiles
        #        X, Y = make_gaussian_quantiles(n_samples=120, n_features=3, n_classes=2)

        #        xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
        #        np.linspace(-3, 3, 50))
        #        rng = np.random.RandomState(0)
        #        X = rng.randn(200, 2)
        #        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

        return X, Y
    # %%
    elif data_set_type == 'multi class':
        """
        Number of targets: 1
        Cardinality: >2

        Within sckit-learn, all estimators supporting binary classification
        also support multiclass classification, using One-vs-Rest by default.
        """
        predict_for_single_row = [5]
        pass
    # %%
    elif data_set_type == 'multi label':
        """
        Number of targets: >1
        Cardinality: 2  (0 and 1)
        """
        from sklearn.datasets import make_multilabel_classification
        predict_for_single_row = [1, 0, 0, 1]
        pass
    # %%
    elif data_set_type == 'multioutput-multiclass classification':
        """
        Number of targets: >1
        Cardinality: >2
        """
        predict_for_single_row = [12, 10, 5, 1]
        pass


# %%

"""
    NOT BINARY CLASSIFICATION DATASETS

    not all

    #from sklearn.datasets import fetch_20newsgroups_vectorized
    #        bunch = fetch_20newsgroups_vectorized(subset="all")
    #        X,Y = bunch.data, bunch.target


    #from sklearn.datasets import load_iris
    #        X, Y = load_iris(return_X_y=True)


    #from sklearn.datasets import load_digits
    #        X, Y = load_digits(return_X_y=True)


    #from sklearn.datasets import load_wine
    #        X, Y = load_wine(return_X_y=True)


    from sklearn.datasets import fetch_kddcup99
    X, Y = fetch_kddcup99(return_X_y=True)


    from sklearn.datasets import fetch_covtype
    X, Y = fetch_covtype(return_X_y=True)


    from sklearn.datasets import fetch_rcv1
    X, Y = fetch_rcv1(return_X_y=True)

"""


# %%    For Benchmarking

def load_benchmark_binary(data_set_type='binary'):
    ds = []

    from sklearn.datasets import load_breast_cancer, make_hastie_10_2, \
        make_moons, make_circles, make_classification, make_blobs, \
        make_gaussian_quantiles

    # %%
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
    array = df.values
    X = array[:, 0:8]
    Y = array[:, 8]
    ds.append(['pima-indians-diabetes', (X, Y)])

    ds.append(['breast_cancer', load_breast_cancer(return_X_y=True)])

    # %%
    ds.append(['make_hastie_10_2', make_hastie_10_2(n_samples=12000)])
    ds.append(['make_hastie_10_2_low', make_hastie_10_2(n_samples=800)])

    ds.append(['make_moons', make_moons(n_samples=100, noise=0.3)])
    ds.append(['make_moons_more', make_moons(n_samples=1000, noise=0.4)])

    ds.append(['make_circles', make_circles(n_samples=100, noise=0.2, factor=0.5)])
    ds.append(['make_circles_more', make_circles(n_samples=2000, noise=0.4, factor=0.5)])

    ds.append(['make_classification', make_classification(n_repeated=0,
                                                          n_classes=2, n_samples=1000, n_features=0 + 3, n_redundant=0,
                                                          n_informative=3, n_clusters_per_class=2)])

    #    genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
    #                               'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
    #                               sep='\t', compression='gzip')
    #    X, Y = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
    #    ds.append(['GAMETES_Epistasis',(X,Y) ])  #AttributeError: 'NuSVC' object has no attribute 'shape_fit_'

    ds.append(['make_blobs', make_blobs(n_samples=120, n_features=5, centers=2)])
    ds.append(['make_blobs_more', make_blobs(n_samples=620, n_features=10, centers=2)])

    ds.append(['make_gaussian_quantiles_more', make_gaussian_quantiles(n_samples=1000, n_features=8, n_classes=2)])
    ds.append(['make_gaussian_quantiles', make_gaussian_quantiles(n_samples=120, n_features=3, n_classes=2)])

    xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                         np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(200, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    ds.append(['XOR', (X, Y)])

    """
    from sklearn.datasets import fetch_openml
    https://scikit-learn.org/stable/datasets/index.html#openml
    """

    return ds


# %%

if __name__ == "__main__":
    # X,Y=load_example_dataset(data_set_type='binary') # old test

    import config

    conf = config.load_config()
    ds_abs_path = conf['Paths']['Abs path to Dataset']
    ds_type = conf['Global variables']['DS_type']
    cd_abs_path = conf['Paths']['Abs path to CD']
    cd_type = conf['Global variables']['CD_type']

    # X, y = data.load_X_y(ds_abs_path, ds_type, cd_abs_path, cd_type)
    print(load_CD(cd_abs_path, cd_type))

    print(load_dataset_numpy(ds_abs_path, ds_type))
    print(type(load_dataset_numpy(ds_abs_path, ds_type)))

# %%

"""
sparce swmlight datasets
#        from sklearn.datasets import load_svmlight_files
##        X_train, y_train = load_svmlight_file("/datasets/train_dataset.txt")
#        X_train, y_train, X_test, y_test = load_svmlight_files(
#                (".//datasets//a1a", "./datasets//a1a.t"))
#        print(type(X_train))
#        print(type(y_train))
#        print(X_train, y_train, X_test, y_test)
#        X,Y=X_train+X_test, y_train+y_test
        #https://scikit-learn.org/dev/datasets/index.html#datasets-in-svmlight-libsvm-format
        #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
"""
