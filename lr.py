# coding='utf-8'
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import pickle
import os
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.stats.stats import pearsonr


train_path = "data\\train.csv"
test_path = "data\\test.csv"
solution_path = "solution\\xgboost.csv"

def get_train():
    dump_path = '.\\data\\train.pkl'
    if os.path.exists(dump_path):
        trainSet = pickle.load(open(dump_path, 'rb'))
    else:
        trainSet = pd.read_csv(train_path)
        pickle.dump(trainSet, open(dump_path, 'wb'))
    return trainSet


def get_test():
    dump_path = '.\\data\\test.pkl'
    if os.path.exists(dump_path):
        testSet = pickle.load(open(dump_path, 'rb'))
    else:
        testSet = pd.read_csv(test_path)
        pickle.dump(testSet, open(dump_path, 'wb'))
    return testSet


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
    return(rmse)


if __name__ == '__main__':

    train = get_train()
    test = get_test()
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    # prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
    # prices.hist(figsize=(12.0, 6.0))
    # plt.show()
    train["SalePrice"] = np.log1p(train["SalePrice"])
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())

    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    '''-------------------'''
    # model_ridge = Ridge()
    #
    # alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    # cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
    #             for alpha in alphas]
    # cv_ridge = pd.Series(cv_ridge, index=alphas)
    # cv_ridge.plot()
    # plt.xlabel("alpha")
    # plt.ylabel("rmse")
    # plt.show()
    #
    # print(cv_ridge.min())

    '''------------------------'''
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)
    print(rmse_cv(model_lasso).mean())
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    print(type(coef))
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +
          str(sum(coef == 0)) + " variables")

    # imp_coef = pd.concat([coef.sort_values().head(10),
    #                       coef.sort_values().tail(10)])

    # imp_coef.plot(kind="barh", figsize=(8.0, 10.0))
    # plt.title("Coefficients in the Lasso Model")
    # plt.show()


    '''------------------'''
    dtrain = xgb.DMatrix(X_train, label=y)
    dtest = xgb.DMatrix(X_test)
    # params = {"max_depth": 2, "eta": 0.1}
    # model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
    # model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()

    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)  # the params were tuned using xgb.cv
    model_xgb.fit(X_train, y)

    xgb_preds = np.expm1(model_xgb.predict(X_test))
    lasso_preds = np.expm1(model_lasso.predict(X_test))

    preds = 0.7 * lasso_preds + 0.3 * xgb_preds
    solution = pd.DataFrame({"Id": test.Id, "SalePrice": preds})
    solution.to_csv("solution\\ridge_sol.csv", index=False)
    print("")