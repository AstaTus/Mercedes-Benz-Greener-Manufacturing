# auther = 'chenjunqi'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

class Ensembler(object):
    def __init__(self, n_folds, base_models):
        self.n_folds = n_folds
        self.base_models = base_models
        self.level_one_train = np.zeros((1, 1));
        self.level_one_test = np.zeros((1, 1));
        self.X = None
        self.y = None
        self.T = None


    def fit_predict(self, X, y, T):
        self.X = np.array(X)
        self.y = np.array(y)
        self.T = np.array(T)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=2016)

        self.level_one_train = np.zeros((X.shape[0], len(self.base_models)))
        self.level_one_test = np.zeros((T.shape[0], len(self.base_models)))

        print(len(self.base_models))
        for i, model in enumerate(self.base_models):
            print("i=%d"%i)
            S_test_i = np.zeros((T.shape[0], self.n_folds))

            for j, (train_idx, test_idx) in enumerate(kf.split(self.X)):
                print("j=%d"%j)
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.level_one_train[test_idx, i] = y_pred
                S_test_i[:, j] = model.predict(T)

            self.level_one_test[:, i] = S_test_i.mean(1)

        return self.level_one_train, self.level_one_test

    def stacking(self, stacker_model):
        stacker_model.fit(self.level_one_train, self.y)
        y_pred = stacker_model.predict(self.level_one_test)[:]
        return y_pred