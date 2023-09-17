import tools
import config
import data
import os
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV

def trainAll(Xtrain, Ytrain, Xtest, Ytest, index, method=None, tuning=None):
    # XGBoost Regressor

    #     from xgboost import XGBRegressor as XGBR
    #     reg = XGBR(n_estimators=10).fit(Xtrain, Ytrain)
    #     y_pred = reg.predict(Xtest)
    #     y_pred = [round(each) for each in y_pred]
    # #     print(Ytest, y_pred)
    #     allresult = evaluate(Ytest, y_pred, allresult, index="XGBoost Regressor")
    #     nprint(need_print,repr(allresult))

    # XGBoost Classification
    if method == "XGBC":
        from xgboost import XGBClassifier as XGBC
        #         from sklearn.naive_bayes import GaussianNB
        # weight_y_train = (len(Ytest)-sum(Ytest))/len(Ytest)

        if tuning:
            model = XGBC()
            parameter = {"gamma":(0.1,0.2,0.5,1,1.5,2,3,4,5),"n_estimators":(10,50,100,200),
                         "max_depth":range(3,15),"learning_rate":(0.01,0.02,0.03,0.05,0.1,0.2,0.3),"eval_metric":("auc",)}
            clf = GridSearchCV(model, parameter, cv=3)
            clf.fit(Xtrain, Ytrain)
            print("clf is:", clf, clf.best_params_, clf.best_estimator_, clf.best_score_)
            y_pred = clf.predict(Xtest)

        else:
            model = XGBC(gamma=0.5, n_estimators=100,max_depth=15, learning_rate=0.03)
            model.fit(Xtrain, Ytrain)
            y_pred = model.predict(Xtest)

        # print(Ytest, y_pred)
        config.all_result = tools.evaluate(Ytest, y_pred, config.all_result, config.all_columns, index=index)
        config.nprint(config.need_print, repr(config.all_result))

    if method == "SVM":
        from sklearn.svm import LinearSVC

        model = LinearSVC()
        model.fit(Xtrain, Ytrain)
        y_pred = model.predict(Xtest)
        # print(Ytest, y_pred)
        config.all_result = tools.evaluate(Ytest, y_pred, config.all_result, config.all_columns, index=index)
        config.nprint(config.need_print, repr(config.all_result))
    return config.all_result


def perform(fname):
    print("getting data", "." * 30)
    X, Y = data.get_data(fname)
    print("getting training data", "." * 50)
    Xtrain, Xtest, Ytrain, Ytest = data.get_train_data(X, Y, config.all_result, config.all_columns)
    #     allresult = pd.DataFrame()
    print("training", "." * 70)

    config.all_result = trainAll(Xtrain, Ytrain, Xtest, Ytest, fname, config.run_method, config.tuning)
    print(config.all_result)
    return config.all_result.iloc[-1, -1]


def train():
    for each in os.listdir(config.dirname):
        if each.endswith("arff"):
            eachname = os.path.join(config.dirname, each)
            print(each)
            for fvalue_thres in range(0,1):
                config.fvalue_thres = fvalue_thres
                config.all_result = config.all_result.append(pd.DataFrame([[each]],columns=["data"]))
                config.all_result.iloc[-1:,config.all_columns.index("method")] = config.run_method
                config.all_result.iloc[-1:,config.all_columns.index("fvalue_thres")] = config.fvalue_thres

                perform(eachname)

# def train_with_bo(fvalue_thres):
#     for each in os.listdir(config.dirname):
#         if each.endswith("arff"):
#             eachname = os.path.join(config.dirname, each)
#             print(each)
#             config.fvalue_thres = fvalue_thres
#             config.all_result = config.all_result.append(pd.DataFrame([[each]],columns=["data"]))
#             config.all_result.iloc[-1:,config.all_columns.index("method")] = config.run_method
#             config.all_result.iloc[-1:,config.all_columns.index("fvalue_thres")] = config.fvalue_thres
#
#     return perform(eachname)

# def bo():
#     pbound = {"fvalue_thres":(1,31)}
#     optimizer = BayesianOptimization(
#         f=train_with_bo,
#         pbounds=pbound)
#     optimizer.maximize(init_points=2,n_iter=10)
#     print(optimizer.max)
def createname():
    name = "result.csv"
    if config.need_fvalue_filter:
        name = "fvalue-" + config.fvalue_thres + "-" + name
    if config.need_undersampleing:
        name = "unsample-" + config.undersample_rate + "-" + name
    if config.need_smote_augument:
        name = "smoteaug-" + name
    return name

if __name__ == "__main__":
    # bo()
    train()
    config.all_result.to_csv("result.csv")