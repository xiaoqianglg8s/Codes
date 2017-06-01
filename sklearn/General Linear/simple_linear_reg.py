import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes=datasets.load_diabetes()

x=diabetes.data[:,np.newaxis,2]

x_train = x[:-20]
x_test = x[-20:]

y_train=diabetes.target[:-20]
y_test=diabetes.target[-20:]

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

print('Coeff:',model.coef_)

print('Mean Sq err:',np.mean((model.predict(x_test)-y_test)**2))

plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,model.predict(x_test),color='blue')

plt.xticks(())
plt.yticks(())