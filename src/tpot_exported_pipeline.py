import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from preprocess import get_data

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
X_train, y_train, X_test = get_data()

# Average CV score on the training set was: -0.12046717668718272
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=50.0)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=0.1)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="ls", max_depth=7, max_features=0.3, min_samples_leaf=6, min_samples_split=7, n_estimators=100, subsample=0.45)),
    LinearSVR(C=0.01, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)
result = pd.DataFrame(y_pred)
result.to_csv('result.txt', index=False, header=False)
