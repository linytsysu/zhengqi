#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocess import get_data
from tpot import TPOTRegressor

if __name__ == "__main__":
    X_train, y_train, X_test = get_data()

    pipeline_optimizer = TPOTRegressor(generations=100, population_size=100, cv=5,
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
