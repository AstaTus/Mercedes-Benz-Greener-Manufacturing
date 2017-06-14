# auther = 'chenjunqi'
# -*- coding: utf-8 -*-

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import numpy as np

class CVHelper:
    # def __init__(self):

    @staticmethod
    def KFoldCV(model, num_fold, X_train, y_train, eval_func, random_state = 100):
        cv_results = []
        kf = KFold(n_splits=num_fold, random_state=random_state)
        for train_index, test_index in kf.split(X_train):
            X_sub_train = X_train[train_index]
            X_sub_test = X_train[test_index]
            y_sub_train = y_train[train_index]
            y_sub_test = y_train[test_index]

            model.fit(X_sub_train, y_sub_train)
            y_pred = model.predict(X_sub_test)

            eval = eval_func(y_sub_test, y_pred)
            cv_results.append(eval)
        eval_mean =  np.mean(cv_results)
        eval_std = np.std(cv_results)
        print('eval mean:%f eval std:%f'%(eval_mean, eval_std))

        return eval_mean, eval_std


    @staticmethod
    def LooCV(model, X_train, y_train, eval_func, silent=False, step=100):
        y_tests = []
        y_preds = []
        i = 0
        loo = LeaveOneOut()
        for train_index, test_index in  loo.split(X_train):
            X_sub_train = X_train[train_index]
            X_sub_test = X_train[test_index]
            y_sub_train = y_train[train_index]
            y_sub_test = y_train[test_index]

            model.fit(X_sub_train, y_sub_train)
            y_pred = model.predict(X_sub_test)

            y_tests.extend(y_sub_test)
            y_preds.extend(y_pred)

            if silent == False:
                i = i + 1;
                if i % step == 0:
                    print(i)

        eval = eval_func(y_sub_test, y_pred)
        print('eval:%f'%(eval))
        return eval
