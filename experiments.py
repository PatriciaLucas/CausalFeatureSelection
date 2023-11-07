import sys
sys.path.append('./')
import time
import numpy as np
import pandas as pd
import models_functions as md
import graph_functions as gf
import datasets_functions as df
import save_database as sd
from os import walk


def experiment(name_dataset, model_name, data, target, flag, database_path, graph_path, methods):

        sd.execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, model_name TEXT, method_graph TEXT, graph_size TEXT, window INT, max_lags BLOB, \
                    step_ahead INT, time_graph FLOAT, yhat BLOB, y_test BLOB, nrmse FLOAT)", database_path)
        
        if name_dataset == 'DOWJONES.csv': 
            dataset = data[:4000]
        elif name_dataset == 'EVAPOTRANSPIRATION.csv' or name_dataset == 'PRSA.csv' or name_dataset == 'SONDA.csv' or name_dataset == 'HOME.csv':
            dataset = data[:10000]
        elif name_dataset == 'XINGU.csv':
            dataset = data[:300]
            
        max_lags = 10

        
        for method_graph in methods:
            #Verifica se o grafo já existe:
            if flag:
                print("Criando o grafo..",method_graph)
                start_time = time.time()
                if method_graph == 'PCMCI':
                    graph = gf.create_causal(dataset, target, max_lags)
                elif method_graph == 'Correlational':
                    graph = gf.create_correlation(dataset, 0.01, target, max_lags)
                elif method_graph == 'GA':
                    graph = gf.create_genetic(dataset, target, max_lags)
                else:
                    graph = gf.create_lasso(dataset, target, max_lags)
                    
                time_graph = time.time() - start_time
                graph.to_csv('C:/Users/Patricia/OneDrive/Área de Trabalho/PROJETO CAUSAL/experimentos_2/g/'+method_graph+'_'+name_dataset, index=False)
                graph_size = graph.to_numpy().sum()
                max_lags = graph.shape[0]
                print("Save")

            
            
            X, y = df.organize_dataset(dataset, graph, max_lags, 1, target)
            if model_name == 'RF':
                best_params = md.gridsearch_RF(X, y)

            
            if name_dataset == 'DOWJONES.csv': 
                dataset = data[4000:]
            elif name_dataset == 'EVAPOTRANSPIRATION.csv' or name_dataset == 'PRSA.csv' or name_dataset == 'SONDA.csv' or name_dataset == 'HOME.csv':
                dataset = data[10000:20000]
            elif name_dataset == 'XINGU.csv':
                dataset = data[200:]
            

            dataset.index =  range(0,dataset.shape[0])
            for step_ahead in [1,3,7]:
                print("Start....")
                X, y = df.organize_dataset(dataset, graph, max_lags, step_ahead, target)
                print(step_ahead)
                
                train_size = 0.8*dataset.shape[0]/3
                test_size = (dataset.shape[0] - 0.8*dataset.shape[0])/3
                
                window = np.arange(0, dataset.shape[0], dataset.shape[0]/3)
                
                janela = 1
            
                for w in window:
                    
                    X_train = X.loc[w:w+train_size]
                    X_test = X.loc[w+train_size:w+train_size+test_size]
                    
                    y_train = y.loc[w:w+train_size]
                    y_test = y.loc[w+train_size:w+train_size+test_size]
                    
                    if model_name == 'RF':
                        #Treinamento
                        model = md.fit_RF(X_train, y_train, best_params)
                    
                        #Teste
                        yhat, nrmse = md.predict_RF(X_test, y_test, model, step_ahead)
                        
                    
        
                    #Salva no banco de dados
                    sd.execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                      (name_dataset, model_name, method_graph, str(graph_size), janela, \
                                        max_lags, step_ahead, time_graph, y_test.to_numpy().tostring(),
                                        yhat.tostring(), nrmse), 
                                      database_path)
                    print("Save: ",name_dataset,'_',method_graph,'_',step_ahead)
                    janela = janela +1
        return

name_dataset = []
for (dirpath, dirnames, filenames) in walk("C:/Users/Patricia/OneDrive/Área de Trabalho/PROJETO CAUSAL/datasets/datasets/"):
    name_dataset.extend(filenames)
    
name_dataset = ['DOWJONES.csv', 'HOME.csv', 'EVAPOTRANSPIRATION.csv', 'PRSA.csv','SONDA.csv', 'XINGU.csv']
methods = ['PCMCI', 'GA', 'Correlational', 'LASSO']
target = ['AVG', 'use', 'ETO', 'PM2.5', 'glo_avg','maxima']
model_name = ['RF']

flag = True
database_path = 'C:/Users/Patricia/OneDrive/Área de Trabalho/PROJETO CAUSAL/experimentos_2/database/bd_5.db'
graph_path = 'C:/Users/Patricia/OneDrive/Área de Trabalho/PROJETO CAUSAL/experimentos_2/graphs/'


for i in range(len(name_dataset)):
    print(name_dataset[i])
    dataset = pd.read_csv(dirpath+name_dataset[i], on_bad_lines='skip', encoding='unicode_escape')
    #dataset = dataset.drop('data', axis = 1)
    #dataset = dataset.drop('J', axis = 1)
    dataset = dataset.interpolate(method='polynomial', order=2)
    experiment(name_dataset[i], model_name[0], dataset, target[i], flag, database_path, graph_path, methods)
