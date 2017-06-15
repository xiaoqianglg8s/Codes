import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rs=np.random.RandomState(0)

boston=load_boston()

x,y=boston.data,boston.target

n_samples=x.shape[0]
n_features=x.shape[1]

rfr=RandomForestRegressor(random_state=0,n_estimators=100)

score=cross_val_score(rfr,x,y)
print("entire data",score)