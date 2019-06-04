#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def save_prediction(y_pred):
    pd.DataFrame(y_pred).to_csv('prediction.txt', index=False, header=False)
