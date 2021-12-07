#%%
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import acf, q_stat, adfuller 
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT
import warnings
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm, t
from thesis_data import dataloader
import matplotlib.pyplot as plt
import pmdarima as pm
import tikzplotlib

np.random.seed(1993)

def VaR_garch(data, alpha, split_date):
    # multiply data by factor 100 to ensure MLE convergence
    garch_model = arch_model(100*data, vol = 'Garch', p = 1, o = 0, q = 1, dist='t', mean='constant')
    # fit model
    res = garch_model.fit(last_obs = split_date, disp='off')
    # get forecasts
    forecasts = res.forecast(horizon = 1, start = split_date, reindex = False)
    mean = forecasts.mean[split_date:]
    var = forecasts.variance[split_date:]
    # compute VaR
    VaR = -(mean.values + np.sqrt(var).values * t.ppf(q=alpha, df=6.2192)) 

    return 0.01*VaR



