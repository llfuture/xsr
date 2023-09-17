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


        if tuning == 'grid':
            model = XGBC()
            parameter = {"scale_pos_weight":range(1,10),"n_estimators":(10,100,1000),"max_depth":range(3,15)}
            clf = GridSearchCV(model, parameter, cv=3)
            clf.fit(Xtrain, Ytrain)
            print("clf is:", clf, clf.best_params_, clf.best_estimator_, clf.best_score_)
            y_pred = model.predict(Xtest)

        elif tuning == 'bay':
            model = XGBC(gamma=0.5, n_estimators=200, max_depth=15, learning_rate=0.01)
            # model = XGBC()
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


def train_with_bo():
    for each in os.listdir(config.dirname):
        if each.endswith("arff"):
            eachname = os.path.join(config.dirname, each)
            print(each)

            config.best_result = config.best_result.append(pd.DataFrame([[each]], columns=["data"]))
            config.best_result.iloc[-1:, config.all_columns.index("method")] = config.run_method

            def perform(fvalue_thres):
                print("getting data", "." * 30)
                config.fvalue_thres = fvalue_thres

                config.all_result = config.all_result.append(pd.DataFrame([[each]], columns=["data"]))
                config.all_result.iloc[-1:, config.all_columns.index("method")] = config.run_method
                config.all_result.iloc[-1:, config.all_columns.index("fvalue_thres")] = config.fvalue_thres

                X, Y = data.get_data(eachname)
                print("getting training data", "." * 50)
                Xtrain, Xtest, Ytrain, Ytest = data.get_train_data(X, Y, each, config.all_result, config.all_columns)
                print("training", "." * 70)
                # score = perform((Xtrain, Xtest, Ytrain, Ytest), each)
                # config.fvalue_thres = fvalue_thres
                config.all_result = trainAll(Xtrain, Ytrain, Xtest, Ytest, each, config.run_method, config.tuning)
                print(config.all_result)
                return config.all_result.iloc[-1, -1]

            if config.need_fvalue_filter:
                pbound = {"fvalue_thres": (1, 31)}
                optimizer = BayesianOptimization(
                    f=perform, pbounds=pbound)

                optimizer.maximize()
                print(optimizer.max)

                config.best_result.iloc[-1:, config.all_columns.index("fvalue_thres")] = optimizer.max['params']['fvalue_thres']
                config.best_result.iloc[-1:, config.all_columns.index("gmean")] = optimizer.max['target']
            else:
                perform(0)

    # return perform(eachname)
def createname(name = "result.csv"):
    if config.need_fvalue_filter:
        name = "dynamic_fvalue-" + name
    if config.need_undersampleing:
        name = "unsample-" + str(config.undersample_randomstate) + "-" + name
    if config.need_smote_augument:
        name = "smoteaug-" + name
    return name

if __name__ == "__main__":
    train_with_bo()
    # train()
    name = createname()
    config.all_result.to_csv(name)
    config.best_result.to_csv(createname("best_result.csv"))