import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm

iris=datasets.load_iris()
x,y=iris.data,iris.target
x=x[y!=0,:2]
y=y[y!=0]

n=len(x)

np.random.seed(0)
order=np.random.permutation(n)

x=x[order]
y=y[order].astype(np.float)

m=int(.9*n)

x_train=x[:m]
y_train=y[:m]
x_test=x[m:]
y_test=y[m:]

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    model=svm.SVC(kernel=kernel,gamma=10)
    model.fit(x_train,y_train)
    
    plt.figure(fig_num)
    plt.clf()
    plt.scatter(x[:,0],x[:,1],c=y,zorder=10,cmap=plt.cm.Paired)
    
    plt.scatter(x_test[:, 0], x_test[:, 1], s=80, facecolors='none', zorder=10)
    
    x_min=x[:,0].min()
    x_max=x[:,0].max()
    y_min=x[:,1].min()
    y_max=x[:,1].max()
    
    xx,yy=np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
    z=model.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    
    plt.pcolormesh(xx,yy,z>0,cmap=plt.cm.Paired)
    plt.contour(xx,yy,z,colors=['k','k','k'],linestyle=['--','-','--'],levels=[-.5,0,.5])
    plt.title(kernel)

    
plt.show()