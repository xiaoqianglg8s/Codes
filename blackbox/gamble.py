import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import pandas as pd

gamblers=100

def casino(win_rate,win_once=1,loss_once=1,commission=0.01):
    my_money=1000000
    play_cnt=10000000
    commission=commission
    for _ in np.arange(0,play_cnt):
        w=np.random.binomial(1,win_rate)
        if w:
            my_money+=win_once
        else:
            my_money-=loss_once
        my_money-=commission
        if my_money<=0:
            break
    return my_money

casino_nb=nb.jit(casino)

#heaven_moneys = [casino_nb(0.5, commission=0) for _ in np.arange(0, gamblers)]
#moneys_low = [casino_nb(0.4, commission=0) for _ in np.arange(0, gamblers)]
#moneys_commission = [casino_nb(0.5, commission=0.01) for _ in np.arange(0, gamblers)]
#moneys_low_commission = [casino_nb(0.4, commission=0.01) for _ in np.arange(0, gamblers)]

#plt.setp(plt.gca().get_xticklabels(), rotation=30)
#pd.Series(heaven_moneys).hist(bins=30)
#pd.Series(moneys_low).hist()
#pd.Series(moneys_commission).hist()

#moneys = [casino_nb(0.5, commission=0.01, win_once=1.02, loss_once=0.98) for _ in np.arange(0, gamblers)]
#plt.setp(plt.gca().get_xticklabels(), rotation=30)
#pd.Series(moneys).hist()

moneys = [casino_nb(0.45, commission=0.01, win_once=1.02, loss_once=0.98) for _ in np.arange(0, gamblers)]
pd.Series(moneys).hist()