import xgboost as xgb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.grid_search import GridSearchCV

def ModelFit(alg, feature_names, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5,
             early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print("\n cv_result")
        print(cvresult)
        print('num_boost_round:%d'%(len(cvresult)))
    # Fit the algorithm on the data
    alg.fit(X_train, y_train)

    # Predict training set:
    y_predictions = alg.predict(X_test)
    #     dtrain_predprob = alg.predict_proba(y_test)

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % metrics.mean_squared_error(y_test, y_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp = feat_imp.rename(lambda x: feature_names[int(x[1:len(x)])])

    feat_imp.plot(kind='barh', title='Feature Importances', figsize=(7, 16))
    plt.ylabel('Feature Importance Score')


def ModelParamSearch(xgb, params, X_train, y_train, score):
    search = GridSearchCV(estimator=xgb, param_grid=params, n_jobs=4, iid=False, cv=5, scoring=score)
    search.fit(X_train, y_train)
    print('\ngrid_scores')
    for score in search.grid_scores_:
        print(score)
    print('\nbest_params')
    print(search.best_params_)
    print('\nbest_score')
    print(search.best_score_)
    return search