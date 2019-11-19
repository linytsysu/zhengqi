#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

class Model(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def fit_eval(self, X_train, y_trian, X_eval, y_eval):
        pass

    @abc.abstractclassmethod
    def fit(self, X_train, y_train):
        pass

    @abc.abstractclassmethod
    def predict(self, X_test):
        pass

    @abc.abstractclassmethod
    def get_submission(self, X_test):
        pass
