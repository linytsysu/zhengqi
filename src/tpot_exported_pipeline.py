import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from preprocess import get_data
from sklearn.model_selection import KFold, cross_val_score


def rmsle_cv(model=None, X_train=None, y_train=None):
    seed = 2019
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train)
    rmse= -cross_val_score(model, X_train, y_train,
                           scoring="neg_mean_squared_error", cv = kf)
    return (rmse)


def get_tpot_pipeline():
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
        StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=0.1)),
        StandardScaler(),
        StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=11, min_samples_split=6, n_estimators=100)),
        LinearSVR(C=0.01, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.01)
    )

    set_param_recursive(exported_pipeline.steps, 'random_state', 42)

    return exported_pipeline


if __name__ == "__main__":
    X_train, y_train, X_test = get_data()

    exported_pipeline = get_tpot_pipeline()

    score = rmsle_cv(exported_pipeline, X_train, y_train)
    print('\nAveraged Models Score: %6f, %6f\n'%(np.mean(score), np.std(score)))

    exported_pipeline.fit(X_train, y_train)
    y_pred = exported_pipeline.predict(X_test)
    result = pd.DataFrame(y_pred)
    result.to_csv('result.txt', index=False, header=False)