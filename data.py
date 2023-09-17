import config
import numpy as np
import tools
import arff
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as TTS
## 读取ppi数据
def get_data(fname='data/ppi/GO_PPI_CE-BP+CC+MF-threshold-3.arff'):
    d = arff.load(fname)
    data = pd.DataFrame(d)
    #     print(data)
    data_object = data.select_dtypes(np.object)
    data_float = data.select_dtypes(np.float64)
    Y = data_object.iloc[:, -1].values.astype(np.int32)
    go = data_object.iloc[:, 1:-1].values.astype(np.float32)
    ppi = data_float.values
    print("go shape is:", go.shape, ",ppi shape is:", ppi.shape, ", Y shape is:", Y.shape)
    if config.need_data_standard:
        sc_X = StandardScaler()
        ppi = sc_X.fit_transform(ppi)
        go = sc_X.fit_transform(go)

    if config.need_fvalue_filter:
        ppi, Y = tools.fvalue_filter(ppi, Y, config.fvalue_thres)
        print("after fvalue filter, go shape is:", go.shape, ",ppi shape is:", ppi.shape, ", Y shape is:", Y.shape)


    X = np.concatenate((go, ppi), axis=1)
    config.all_result.iloc[-1:, config.all_columns.index("x_fstat_shape")] = str(X.shape)

    if config.only_go:
        X = go
    if config.only_ppi:
        X = ppi

    if config.need_usepca:
        pca = PCA(n_components=10)
        X = pca.fit_transform(X)
        print("pca ratio is:", pca.explained_variance_ratio_)

    if config.need_data_standard:
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)

    print("x_shape=", X.shape, ",y_shape=", Y.shape)
    return X, Y

def save_train_data(xtrain, ytrain, fname, sheetname="origin"):
    fname = "final_data/" + fname[:fname.rfind(".")] + "_" + sheetname + ".csv"
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xy = np.concatenate((xtrain, ytrain.reshape([-1,1])), axis=1)
    xy_dataframe = pd.DataFrame(xy)
    # writer = pd.ExcelWriter(fname)
    xy_dataframe.to_csv(fname)

## 随机分割数据为训练数据和测试数据
def get_train_data(X, Y, fname, all_result, all_columns):
    Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=0.1, random_state=3)
    # print(Xtrain, Ytrain)
    if config.store_undersample_data or config.store_oversample_data:
        save_train_data(Xtrain, Ytrain, fname, "origin_train")
        save_train_data(Xtest, Ytest, fname, "origin_test")
    all_result.iloc[-1:,all_columns.index("x_train_shape")] = str(Xtrain.shape)
    all_result.iloc[-1:,all_columns.index("y_train_shape")] = str(Ytrain.shape)
    all_result.iloc[-1:,all_columns.index("x_test_shape")] = str(Xtest.shape)
    all_result.iloc[-1:,all_columns.index("y_test_shape")] = str(Ytest.shape)
    all_result.iloc[-1:, all_columns.index("one_class_ratio_train")] = str(sum(Ytrain) / len(Ytrain))
    all_result.iloc[-1:, all_columns.index("one_class_ratio_test")] = str(sum(Ytest) / len(Ytest))

    config.nprint(True, "before augment class 1 ratio in Ytrain is %f and in Ytest is %f" % (
    sum(Ytrain) / len(Ytrain), sum(Ytest) / len(Ytest)))

    if config.need_undersampleing:
        Xtrain, Ytrain = tools.under_sampling(Xtrain, Ytrain)
        if config.store_undersample_data:
            save_train_data(Xtrain, Ytrain, fname, "undersample_train")
        # Xtrain, Ytrain = tools.under_sampling_cluster(Xtrain, Ytrain)
    # config.nprint(config.need_print, "x_train_shape=", Xtrain.shape, ",x_test_shape=", Xtest.shape, ",y_train_shape=", Ytrain.shape,
    #        ",y_test_shape=", Ytest.shape)
        all_result.iloc[-1:, all_columns.index("one_class_ratio_train_undersample")] = str(sum(Ytrain) / len(Ytrain))
        config.nprint(True, "after undersample class 1 ratio in Ytrain is %f and in Ytest is %f" % (
        sum(Ytrain) / len(Ytrain), sum(Ytest) / len(Ytest)))

    if config.need_smote_augument:
        Xtrain, Ytrain = tools.data_smote_augument(Xtrain, Ytrain)

    if config.need_bsmote_augument:
        Xtrain, Ytrain = tools.data_bsmote_augument(Xtrain, Ytrain)
        if config.store_oversample_data:
            save_train_data(Xtrain, Ytrain, fname, "oversample_train")
        all_result.iloc[-1:, all_columns.index("one_class_ratio_train_oversample")] = str(sum(Ytrain) / len(Ytrain))

    if config.need_adasyn_augument:
        Xtrain, Ytrain = tools.data_adasyn_augument(Xtrain, Ytrain)

    if config.need_ksmote_augument:
        Xtrain, Ytrain = tools.data_ksmote_augument(Xtrain, Ytrain)

    # config.nprint(config.need_print, "x_train_shape=", Xtrain.shape, ",x_test_shape=", Xtest.shape, ",y_train_shape=",
    #               Ytrain.shape,
    #               ",y_test_shape=", Ytest.shape)
    config.nprint(True, "after augment class 1 ratio in Ytrain is %f and in Ytest is %f" % (
        sum(Ytrain) / len(Ytrain), sum(Ytest) / len(Ytest)))



    # all_result.iloc[-1:, all_columns.index("one_class_ratio_train_oversample")] = str(Xtrain.shape)
    return Xtrain, Xtest, Ytrain, Ytest