# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:46:31 2023

@author: Patricia
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from genetic_selection import GeneticSelectionCV
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests import ParCorr#, GPDC, CMIknn, CMIsymb


def significance_ccf(target, y):
    from scipy.stats import ttest_1samp
    pvalues = []
    num_amostras = 1000
    corr_orig = sm.tsa.stattools.ccf(target, y)

    permuted = np.empty((len(target), num_amostras))
    permuted_corr = np.empty((len(target), num_amostras))

    for i in range(num_amostras):
        permuted[:,i] = np.random.permutation(target)
        permuted_corr[:,i] = sm.tsa.stattools.ccf(target, permuted[:,i])
        
    for i in range(len(target)):
        mean = np.mean(permuted_corr[i][:])
        dp = np.std(permuted_corr[i][:])
        distribution = np.random.normal(loc=mean, scale=dp, size=1000)
        stat, pvalue = ttest_1samp(distribution, corr_orig[i])
        pvalues.append(np.round(pvalue,5))
    return np.array(pvalues)

def create_correlation(dataset, conf_level, target, max_lags):
    G = pd.DataFrame(np.nan, index = np.arange(0,max_lags), columns = dataset.columns.values)
    for n in dataset.columns.values:
        if n != target:
            pvalue = significance_ccf(dataset[target][:max_lags].values, dataset[n][:max_lags].values)
            G[n].loc[:] = np.where(pvalue <= conf_level, True, False)
        else:
            pvalue = sm.tsa.stattools.acf(dataset[target].values, qstat=True)[2][:max_lags]
            G[n].loc[:] = np.where(pvalue <= conf_level, True, False)
    G.index = range(1,G.shape[0]+1)
    return G


def organize_causal(G, var_names, target):
    new_G = pd.DataFrame(columns=list(range(0,G.shape[2])))
    id_target = var_names.index(target)
    for var in range(len(var_names)):
        new_G.loc[var] = list(G[var][id_target] == "-->")
    new_G.index = var_names
    return new_G.T.iloc[1:]

def create_causal(dataset, alpha_level, target, max_lags):
    var_names = list(dataset.columns.values)
    
    for var in range(len(var_names)):
        if var == 0:
            data = dataset[var_names[var]].values.reshape((-1,1))
        else:
            if len(var_names) > 1:
                data = np.concatenate((data,dataset[var_names[var]].values.reshape((-1,1))), axis=1)
            else:
                break

    dataframe = pp.DataFrame(data, datatime = {0:np.arange(len(dataset))}, var_names=var_names)
    
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_lags, pc_alpha=0.1, alpha_level=alpha_level)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=max_lags, fdr_method='fdr_bh')
    G = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=alpha_level, tau_min=0, tau_max=max_lags, selected_links=None)
    G_list = []
    for var in var_names:
        G_list.append(organize_causal(G, var_names, var))
    id_target = var_names.index(target)
    return G_list[id_target]

def organize_genetic(dataset, target, max_lags):
    import itertools
    cols = []
    for i in range(max_lags):
        cols.append(dataset.columns.values)
        
    c = list(itertools.chain.from_iterable(cols))
    X = pd.DataFrame(columns=c)
    for row in range(dataset.shape[0]-max_lags):
        X.loc[X.shape[0],:] = np.reshape(dataset.loc[row:row+max_lags-1].values,(1,dataset.shape[1]*max_lags),order='F')
    y = dataset[target].loc[max_lags:]
    return X, y
  
def create_genetic(dataset, target, max_lags):
    X, y = organize_genetic(dataset, target, max_lags)
    estimator = LinearRegression()
    selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=0,
        scoring="neg_root_mean_squared_error",
        max_features=max_lags,
        n_population=80,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=100,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=10,
        caching=True,
        n_jobs=1,
    )
    selector = selector.fit(X, y)

    G = pd.DataFrame(np.nan, index = np.arange(0,max_lags), columns=dataset.columns.values)
    j = 0
    for n in dataset.columns.values:
        G[n].loc[:] = selector.support_[j:max_lags+j]
        j = j + max_lags
    G.index = range(1,G.shape[0]+1)
    return G
    


def create_lasso(dataset, target, max_lags):
    X, y = organize_genetic(dataset, target, max_lags)
    lsvc = linear_model.Lasso(alpha=0.1).fit(X, y)
    selector = np.where(lsvc.coef_ > 0, True, False)
    G = pd.DataFrame(np.nan, index = np.arange(0,max_lags), columns=dataset.columns.values)
    j = 0
    for n in dataset.columns.values:
        G[n].loc[:] = selector[j:max_lags+j]
        j = j + max_lags
    G.index = range(1,G.shape[0]+1)
    return G
    


















