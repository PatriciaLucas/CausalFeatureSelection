# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:12:30 2023

@author: Patricia
"""
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from sklearn.linear_model import LinearRegression



def fit_LR(X_train, y_train):
  model = LinearRegression().fit(X_train, y_train)
  return model

def predict_LR(X_test, y_test, model, target):
  yhat = model.predict(X_test)
  rmse = math.sqrt(mean_squared_error(y_test,yhat))
  maxmin = np.max(y_test) - np.min(y_test)
  nrmse = rmse/maxmin
  return yhat, rmse, nrmse







