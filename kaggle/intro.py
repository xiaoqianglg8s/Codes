from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

iris=load_iris()
X_iris, y_iris = iris.data, iris.target
X, y = X_iris[:, :2], y_iris

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

from sklearn import metrics
y_train_predict = clf.predict(X_train)
print metrics.accuracy_score(y_train, y_train_predict)

y_predict = clf.predict(X_test)
print metrics.accuracy_score(y_test, y_predict)

print metrics.classification_report(y_test, y_predict, target_names = iris.target_names)