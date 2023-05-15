# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:12:30 2023

@author: Patricia
"""
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.constraints import max_norm, unit_norm
from keras.layers import Dense, Flatten, SpatialDropout1D, Activation, Add, BatchNormalization, LSTM, Input
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping




def fit_LR(X_train, y_train):
  model = LinearRegression().fit(X_train, y_train)
  return model

def predict_LR(X_test, y_test, model, target):
  yhat = model.predict(X_test)
  rmse = math.sqrt(mean_squared_error(y_test,yhat))
  #maxmin = np.max(y_test) - np.min(y_test)
  nrmse = rmse/np.std(y_test)
  return yhat, rmse, nrmse


def fit_LSTM(X_train, y_train):
    individual = {
        'num_layers': 1,
        'units': 72,
        'dropout': 0.3,
        'norm': 0
        } 
    model = Sequential()
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    for i in range(individual['num_layers']):
        if i != individual['num_layers']:
          model.add(LSTM(units=individual['units'], activation='relu', input_shape=(X_train.shape[1],1), return_sequences=True))
        else:
          model.add(LSTM(units=individual['units'], activation='relu', input_shape=(X_train.shape[1],1), return_sequences=False))
        model.add(SpatialDropout1D(round(individual['dropout'],1)))
        if individual['norm'] == 0:
          model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam')
    model.fit(X_train, y_train, epochs = 100, verbose=0, batch_size=32, callbacks = call)
    return model

def predict_LSTM(X_test, y_test, model, target):
  yhat = model.predict(X_test)
  rmse = math.sqrt(mean_squared_error(y_test,yhat))
  #maxmin = np.max(y_test) - np.min(y_test)
  nrmse = rmse/np.std(y_test)
  return yhat, rmse, nrmse







