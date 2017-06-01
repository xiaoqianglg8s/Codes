from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

model=linear_model.LinearRegression()
boston=datasets.load_boston()
y=boston.target

y_=cross_val_predict(model,boston.data,y,cv=10)

fig,ax=plt.subplots()
ax.scatter(y,y_)
ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
ax.set_xlabel('measure')
ax.set_ylabel('predict')
plt.show()