from sklearn import datasets, neighbors, linear_model

digits=datasets.load_digits()
x,y=digits.data,digits.target

n=len(x)

m=int(.9*n)

x_train=x[:m]
y_train=y[:m]
x_test=x[m:]
y_test=y[m:]

knn=neighbors.KNeighborsClassifier()
logistic=linear_model.LogisticRegression()

print(knn.fit(x_train,y_train).score(x_test,y_test))
logistic.fit(x_train,y_train)
s=logistic.score(x_test,y_test)

print(s)