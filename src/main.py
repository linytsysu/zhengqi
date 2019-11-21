#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from preprocess import get_data
from models import build_model, AveragingModels


def rmsle_cv(model=None, X_train=None, y_train=None):
    seed = 2019
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train)
    rmse= -cross_val_score(model, X_train, y_train,
                           scoring="neg_mean_squared_error", cv = kf)
    return (rmse)


if __name__ == "__main__":
    X_train, y_train, X_test = get_data()
    svr, line, lasso, ENet, KRR1, KRR2, lgbm, xgb, nn = build_model()
    averaged_models = AveragingModels(models=(svr, KRR2, ENet, lgbm, xgb))
    score = rmsle_cv(averaged_models, X_train, y_train)

    averaged_models.fit(X_train, y_train)
    y_pred = averaged_models.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv('result.txt', index=False, header=False)

