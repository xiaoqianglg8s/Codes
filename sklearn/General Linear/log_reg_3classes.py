import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

h=.02

model=linear_model.LogisticRegression(C=1e5)

model.fit(x,y)

x_min,x_max = x[:,0].min()-.5,x[:,0].max()+.5
y_min,y_max = x[:,1].min()-.5,x[:,1].max()+.5
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)   
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)            
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xticks(())
plt.yticks(())