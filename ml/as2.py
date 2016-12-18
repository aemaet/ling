import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import cross_validation,datasets,metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

def as1():
	X, y = [],[]
	f = open('wine.data','r')
	for line in f:
		tmp = line.split(',')
		X.append(map(float,tmp[1:]))
		y.append(float(tmp[0]))
	#neigh.fit(X,y)
	X, y = scale(np.array(X)), np.array(y)


	kf = KFold(len(y),n_folds = 5, shuffle = True, random_state = 42)
	res = np.array([])
	t_res = []
	for i in range(1,51):
		neigh = KNeighborsClassifier(n_neighbors=i)
	#	for train_index, test_index in kf:
		#	print train_index, test_index
	#		X_train, X_test = X[train_index], X[test_index]
	#		y_train, y_test = y[train_index], y[test_index]
	#		neigh.fit(X_train,y_train)
	#		#print neigh.score(X_test,y_test)
	#		res = np.append(res,neigh.score(X_test,y_test))
		res = cross_validation.cross_val_score(neigh, X, y, cv=kf)	
		t_res.append(res.mean())
	#for t in t_res:
	#	print t_res.index(t), t
	print t_res
	print t_res.index(max(t_res)), max(t_res)

def as2():
	boston = datasets.load_boston()
	X, y = scale(boston.data), boston.target
	grid = np.linspace(1.,10.,num=200)
	t_res = []
	kf = KFold(len(y),n_folds = 5, shuffle = True, random_state = 42)
	for i in grid:
		neigh = KNeighborsRegressor(n_neighbors=5, weights = 'distance', p = i)
		res = cross_validation.cross_val_score(neigh, X, y, cv=kf,  scoring='mean_squared_error')
		print res
		t_res.append(res.mean())
	print t_res
	print t_res.index(max(t_res)), max(t_res), grid[t_res.index(max(t_res))]

train = pandas.read_csv('perceptron-train.csv')
test = pandas.read_csv('perceptron-test.csv')
x_train, y_train = train.ix[:,1:].as_matrix(),map(int,train.ix[:,0].as_matrix())
x_test, y_test = test.ix[:,1:].as_matrix(),map(int,test.ix[:,0].as_matrix())
pct = Perceptron(random_state=241)
pct.fit(x_train,y_train)
pred = pct.predict(x_test)
a1 = metrics.accuracy_score(pred,y_test)
scaler = StandardScaler()
x_t = scaler.fit_transform(x_train)
x_tst = scaler.transform(x_test)
pct.fit(x_t,y_train)
pred = pct.predict(x_tst)
a2 = metrics.accuracy_score(pred,y_test)
print a1, a2, a2 - a1
