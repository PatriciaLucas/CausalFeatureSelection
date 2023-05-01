# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:34:58 2023

@author: Patricia
"""

import sys
sys.path.append('./')
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import models_functions as md
import graph_functions as gf
import datasets_functions as df
import save_database as sd
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

            

def experiment(name_dataset, dataset, target, params, flag, database_path, graph_path):
    #Criação da tabela no banco de dados:
    sd.execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, method_graph TEXT, graph_size TEXT, window INT, params BLOB, \
            step_ahead INT, time_graph FLOAT, yhat BLOB, y_test BLOB, rmse FLOAT, nrmse FLOAT)", database_path)
    for col in dataset.columns.values:
        p = adfuller(dataset[col].values)[1]
        if p > 0.05:
            dataset[col] = dataset[col].values - dataset[col].shift()
            dataset = dataset.drop(labels=0, axis=0)
    
    data_graph = dataset.loc[:params['size_data_graph']]
    dataset = dataset.loc[params['size_data_graph']:]
    dataset.index = range(1,dataset.shape[0]+1)
    
    params['size_train'] = round((dataset.shape[0] - params['size_test']) / 5) - 1

    for method_graph in ['causal','correlation','genetic','lasso']:
        #Verifica se o grafo já existe:
        if flag:
            print("Criando o grafo..",method_graph)
            start_time = time.time()
            if method_graph == 'causal':
                graph = gf.create_causal(data_graph, params['alpha_level'], target, params['max_lags'])
            elif method_graph == 'correlation':
                graph = gf.create_correlation(data_graph, params['conf_level'], target, params['max_lags'])
            elif method_graph == 'genetic':
                graph = gf.create_genetic(data_graph, target, params['max_lags'])
            else:
                graph = gf.create_lasso(data_graph, target, params['max_lags'])
            time_graph = time.time() - start_time
            graph.to_csv(graph_path+method_graph+'_'+name_dataset, index=False)
            graph_size = graph.to_numpy().sum()
        else:
            #g = np.load(graph_path+'_'+method_graph+'.npy')
            graph = pd.read_csv(graph_path+method_graph+'_'+name_dataset)
            graph = graph.drop(columns=['Unnamed: 0'])
            graph.index = range(1,graph.shape[0]+1)
            time_graph = 0
            graph_size = graph.to_numpy().sum()
        
        for step_ahead in [1,3,5,7,10]:
            print("Start....")
            X, y = df.organize_dataset(dataset, graph, params['max_lags'], step_ahead, target)
            scalerX = MinMaxScaler()
            scalerX.fit(X)
            X_norm = scalerX.transform(X)
            X = pd.DataFrame(X_norm, columns=X.columns.values)
            
            window = np.arange(0, dataset.shape[0], params['size_train'])
            janela = 1
            window = np.delete(window, -1)
            for w in window:
                
                X_train = X.loc[w:w+params['size_train']]
                X_test = X.loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
                
                y_train = y.loc[w:w+params['size_train']]
                y_test = y.loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
                
                #Treinamento
                model = md.fit_LR(X_train, y_train)
                
                #Teste
                yhat, rmse, nrmse = md.predict_LR(X_test, y_test, model, target)

                
                #Salva no banco de dados
                sd.execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (name_dataset, method_graph, str(graph_size), janela, \
                              str(params), step_ahead, time_graph, yhat.tostring(), y_test.to_numpy().tostring(), rmse, \
                              nrmse), database_path)
                print("Save: ",name_dataset,'_',method_graph,'_',step_ahead)
                janela = janela +1
    return


















