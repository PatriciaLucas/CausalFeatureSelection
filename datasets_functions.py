# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 07:40:57 2023

@author: Patricia
"""

import pandas as pd


def get_dataset(data, size_test, max_lags, var_names):
    '''
    Parameters
    ----------
    data : série temporal
    size_test : tamanho da série temporal usada para teste.
    max_lags : número de lags usado para criar o grafo causal.
    var_names : list com o nome das variáveis que constam no grafo causal.
    '''
    data.index = range(0,data.shape[0])
    data = data[var_names]
    X_train = data.loc[:data.shape[0]-size_test]
    #y_train = data.loc[max_lags:data.shape[0]-size_test]
    X_test = data.loc[data.shape[0]-size_test:]
    X_test.index = range(0,X_test.shape[0])
    y_test = data.loc[data.shape[0]-size_test+max_lags:]
    y_test.index = range(0,y_test.shape[0])
    
    return X_train, X_test, y_test

def organize_dataset(dataset, G, max_lags, step_ahead, target):
    lags = G.where(G).stack().index.tolist()
    y = dataset[target].loc[max_lags+step_ahead-1:]
    y.index = range(0,y.shape[0])
    
    for row in range(0,y.shape[0]):
        cols = []
        values = []
        bloco = dataset.iloc[row:max_lags+row]   
        bloco.index = reversed(range(1,bloco.shape[0]+1))
        for lag in lags:
            if row == 0:
                cols.append(lag[1])
                X = pd.DataFrame(columns=cols)
            values.append(bloco[lag[1]].loc[lag[0]])
        
        X.loc[row,:] = values
    return X, y
