import numpy as np
import matplotlib.pyplot as plt
import sklearn
#from sklearn import linear_model, decomposition, datasets
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV

logistic=sklearn.linear_model.LogisticRegression()
pca=sklearn.decomposition.PCA()

pipe=Pipeline([('pca',pca),('logistic',logistic)])

digits=sklearn.datasets.load_digits()

x,y=digits.data,digits.target

pca.fit(x)
plt.figure(1)
plt.plot(pca.explained_variance_,linewidth=2)
plt.xlabel('n_components')
plt.ylabel('explained_var')

n_components=[20,40,64]
Cs=np.logspace(-4,4,3)

estimator=GridSearchCV(pipe,dict(pca__n_components=n_components,logistic__C=Cs))
estimator.fit(x,y)

plt.show()

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

print(estimator.best_estimator_)
