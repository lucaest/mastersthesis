import numpy as np
from scipy.stats import norm, zscore, chi2


def hit_durations(violations):
    duration = []
    count = 1
    for i in violations:
        if i == 0:
            count += 1
        elif i == 1:
            duration.append(count)
            count = 1
    
    return duration

def over_under(loss, VaR):
    n = len(loss)
    over_under = []

    for i in range(0, n):
        if loss[i]>0:
            over_under.append(VaR[i] - loss[i])

    return np.asarray(over_under).flatten()

def Kupiec(alpha, sum_violations, N):
    I = sum_violations
    alpha_hat = I/N
    pof = -2 * np.log(((1 - alpha)**(N - I) * alpha**I)/((1 - alpha_hat)**(N - I) * alpha_hat**I))
    if np.isnan(pof)==True:
        pof = -2 * ((N - I) * np.log(1 - alpha) + I * np.log(alpha) - (N - I) * np.log(1 - alpha_hat) - I * np.log(alpha_hat))
    crit_val = chi2.ppf(1-alpha, df=1)
    p_value = 1-chi2.cdf(pof, df=1)

    if pof < crit_val:
        decision = 'accepted'
    if pof > crit_val:
        decision = 'rejected'

    return decision, round(pof, 5), round(p_value, 5), round(1-alpha_hat, 5)

def Christoffersen(alpha, Hit, sum_violations, N):
    n00, n01, n10, n11 = 0, 0, 0, 0 
    
    for i in range(1, N):
        if (Hit[i] == True and Hit[i - 1] == False):
            n00 += 1
        if (Hit[i] == True and Hit[i - 1] == True):
            n01 += 1
        if (Hit[i] == True and Hit[i - 1] == False):
            n10 += 1
        if (Hit[i] == True and Hit[i - 1] == True):
            n11 += 1 
        
    pi0 = n01/(n00 + n01)
    pi1 = n11/(n10 + n11)
    pi = (n01 + n11)/(n00 + n01 + n10 + n11)
    LRind = -2 * np.log(((1 - pi)**(n00 + n10) * pi**(n01 + n11))/((1 - pi0)**n00 * pi0**n01 * (1 - pi1)**n10 *
        pi1**n11))
    if np.isnan(LRind)==True:
        LRind = -2 * ((n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi) - n00 * np.log(1 - pi0) - n01 *
            np.log(pi0) - n10 * np.log(1 - pi1) - n11 * np.log(pi1))
    LRpof = Kupiec(alpha, sum_violations, N)[1]
    LRcc = LRpof + LRind
    crit_val = chi2.ppf(1-alpha, df=2)
    p_value = 1 - chi2.cdf(LRcc, df = 2)

    if LRcc < crit_val:
        decision = 'accepted'
    if LRcc > crit_val:
        decision = 'rejected'
    
    return decision, round(LRcc, 5), round(p_value, 5)
