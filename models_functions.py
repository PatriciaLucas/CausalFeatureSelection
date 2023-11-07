# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:12:30 2023

@author: Patricia
"""
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

def gridsearch_RF(X, Y):
    params = {
     'bootstrap': [True],
     'min_samples_leaf': [5,10,15,20,25,30],
     'n_estimators': [10,60,110,160,210,260,310,360,410,460,510],
     'max_features': ['sqrt', 'log2'],
     }
    model = GridSearchCV(estimator = ensemble.RandomForestRegressor(), param_grid = params, 
                           cv = 3, n_jobs = -1, verbose = 0)
    grid_result = model.fit(X, Y)
    best_params = grid_result.best_params_
    return best_params


def fit_RF(X_train, y_train, best_params):
    
    model = ensemble.RandomForestRegressor(n_estimators=best_params['n_estimators'], 
                                           min_samples_leaf=best_params['min_samples_leaf'], 
                                           random_state=0)
    model.fit(X_train, y_train)
    return model

def predict_RF(X_test, y_test, model, step_ahead):
  yhat = model.predict(X_test)
  rmse = math.sqrt(mean_squared_error(y_test,yhat))
  maxmin = np.max(y_test) - np.min(y_test)
  nrmse = rmse/maxmin
  return yhat, nrmse





