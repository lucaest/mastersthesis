
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import datetime

def VaR_hist(x, l, alpha):
    N = len(x)
    # sort data
    x_sorted = np.sort(l(x))
    # get worst losses for given alpha level
    VaR_hat = x_sorted[int(N*(1-alpha)+1)]

    return VaR_hat
