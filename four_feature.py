# coding='utf-8'
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn .grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


df_train = get_train()
df_train = pd.concat([df_train['GrLivArea'], df_train['TotalBsmtSF'],
                      df_train['OverallQual'], df_train['GarageCars'], df_train['SalePrice']], axis=1)


X = df_train.ix[:, :-1].values
y = df_train.ix[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

if __name__ == '__main__':
    start = time.clock()
    pipe_xgb = Pipeline([('scl', StandardScaler()),
                         ('xgb', xgb.XGBRegressor( n_estimators=40, max_depth=3, min_child_weight=1,
                                                  subsample=0.9, seed=1))
                         ])
    param_grid = {
        'xgb__reg_alpha': [0.01, 0.1, 1]

    }
    gs = GridSearchCV(estimator=pipe_xgb, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1)
    gs = gs.fit(X, y)
    # print(gs.grid_scores_, '\n', gs.best_params_, '\n', gs.best_score_)
    pipe_xgb = gs.best_estimator_

    # y_train_pred = pipe_xgb.predict(X_train)
    # y_test_pred = pipe_xgb.predict(X_test)
    # # print("MSE train: %.3f, test: %.3f" % (mean_squared_error(y_train, y_train_pred),
    # #                                        mean_squared_error(y_test, y_test_pred)))
    #
    # print("R^2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
    #                                        r2_score(y_test, y_test_pred)))
    #

    df_test = get_test()
    X_test_ = pd.concat([df_test['GrLivArea'], df_test['TotalBsmtSF'],
                          df_test['OverallQual'], df_test['GarageCars']], axis=1)

    X_test_ = X_test_.fillna(0)
    # print(X_test_.isnull().sum())
    y_pred = pipe_xgb.predict(X_test_.values)
    solution = pd.DataFrame({"SalePrice": y_pred, "Id": df_test.Id})
    # print(solution)
    solution.to_csv(solution_path, index=False)
    end = time.clock()
    print('用时：%f s' % (end - start))
