import math
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix as CM
import config

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.ensemble import EasyEnsembleClassifier

def evaluate(ytrue, ypredict, all_result, all_columns,index=None):
#     cr = CR(ytrue, ypredict)
#     print(cr)
    cm = CM(ytrue, ypredict)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    gmean = math.pow(recall * specificity, 0.5)
    f1 = (2* precision * recall) / (precision + recall)
#     result = {"accuracy":[accuracy], "precision":[precision], "recall":[recall],"specificity":[specificity],"fpr":[fpr],"gmean":[gmean],"f1":[f1]}
#     result = pd.DataFrame(result, index=[index])
#     all_result = all_result.append(result)
    all_result.iloc[-1:,all_columns.index("accuracy")] = accuracy
    all_result.iloc[-1:,all_columns.index("precision")] = precision

    all_result.iloc[-1:,all_columns.index("recall")] = recall
    all_result.iloc[-1:,all_columns.index("specificity")] = specificity
    # all_result.iloc[-1:,all_columns.index("fpr")] = fpr
    all_result.iloc[-1:,all_columns.index("gmean")] = gmean
    # all_result.iloc[-1:,all_columns.index("f1")] = f1
    return all_result


## 计算fstatistical
def fvalue_filter(X, Y, fvalue_thres=None):
    # 模型训练
    fvalues = np.array([sm.OLS(Y, X[:, i]).fit().fvalue for i in range(0, X.shape[1])])
    if fvalue_thres == 0:
        itemindex = np.argwhere(fvalues >= fvalues.mean())
    else:
        itemindex = np.argwhere(fvalues >= fvalue_thres)
    # fvalues[fvalues<fvalues.mean()] = 0
    itemindex = np.squeeze(itemindex)

    ##根据fvalues筛选X
    X = X[:, itemindex]

    config.nprint(config.need_print, X.shape, Y.shape)

    return X, Y
#     X[X < 0.01] = 0
#     print(X.shape, Y.shape)



def data_smote_augument(Xtrain, Ytrain):
    oversample = SMOTE(sampling_strategy={1:config.oversample_rate})
    X_, Y_ = oversample.fit_resample(Xtrain, Ytrain)
    return X_, Y_

def data_bsmote_augument(Xtrain, Ytrain):
    oversample = BorderlineSMOTE()
    X_, Y_ = oversample.fit_resample(Xtrain, Ytrain)
    return X_, Y_

def data_adasyn_augument(Xtrain, Ytrain):
    oversample = ADASYN()
    X_, Y_ = oversample.fit_resample(Xtrain, Ytrain)
    return X_, Y_

def data_ksmote_augument(Xtrain, Ytrain):
    oversample = KMeansSMOTE(cluster_balance_threshold=0.02)
    X_, Y_ = oversample.fit_resample(Xtrain, Ytrain)
    return X_, Y_

def under_sampling(Xtrain, Ytrain):
    usampling = RandomUnderSampler(sampling_strategy=config.undersample_rate, random_state=config.undersample_randomstate)
    X_, Y_ = usampling.fit_resample(Xtrain, Ytrain)
    return X_, Y_

def under_sampling_cluster(Xtrain, Ytrain):
    usampling = ClusterCentroids()
    X_, Y_ = usampling.fit_resample(Xtrain, Ytrain)
    return X_, Y_

