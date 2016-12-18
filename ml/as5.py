import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold,cross_val_score, train_test_split 
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pandas

def as1():
	data = pandas.read_csv('abalone.csv')
	data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
	X = data.ix[:,:-1]
	y = data.ix[:,-1]
	cv = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
	for i in range(50):
		clf = RandomForestRegressor(n_estimators=i+1,random_state=1)
		cvs = cross_val_score(clf,X,y,scoring='r2',cv=cv).mean()
		print i+1, cvs

def sigm(y):
	return 1/(1+np.exp(-y))
def as2():
	data = pandas.read_csv('gbm-data.csv')
	X = data.ix[:,1:].as_matrix()
	y = data.ix[:,0].as_matrix()

	test_loss, train_loss = [],[]
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
	                                      test_size=0.8, 
	                                      random_state=241)
	print len(y_test)
	for i in [0.2]  :
		clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
		clf.fit(X_train,y_train)
		#sfd_train = clf.staged_decision_function(X_train)
		#sfd_test = clf.staged_decision_function(X_test)
		for j, y_decision in enumerate(clf.staged_decision_function(X_test)):
			y_pred = 1.0 / (1.0 + np.exp(-y_decision))
			test_loss.append(log_loss(y_test, y_pred))
		print 1
		for j, y_decision in enumerate(clf.staged_decision_function(X_train)):
			y_pred = 1.0 / (1.0 + np.exp(-y_decision))
			train_loss.append(log_loss(y_train, y_pred))
	#	print test_loss,train_loss

	#	yp_test = []
	#	yp_train = []
	#	for j in sfd_train:
	#		yp_train.append(sigm(j))
	#	for j in sfd_test:
	#		print j, len(j)
	#		yp_test.append(sigm(j))
	#	test_loss.append(log_loss(y_test,yp_test))
	#	train_loss.append(log_loss(y_train,yp_train))
	m = min(test_loss)
	print m, test_loss.index(m)
	plt.figure()
	plt.plot(test_loss, 'r', linewidth=2)
	plt.plot(train_loss, 'g', linewidth=2)
	plt.legend(['test', 'train'])
	plt.show()

data = pandas.read_csv('gbm-data.csv')
X = data.ix[:,1:].as_matrix()
y = data.ix[:,0].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	                                      test_size=0.8, 
	                                      random_state=241)
clf = RandomForestClassifier(n_estimators=36,random_state=241)
clf.fit(X_train,y_train)
print log_loss(y_test,clf.predict_proba(X_test))