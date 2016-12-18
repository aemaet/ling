import numpy as np
import pandas
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets, grid_search
from sklearn.metrics import roc_auc_score

def as2():
	newsgroups = datasets.fetch_20newsgroups(data_home='.',
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
	vzer = TfidfVectorizer()
	X = vzer.fit_transform(newsgroups.data)
	y = newsgroups.target
	grid = {'C': np.power(10.0, np.arange(-5, 6))}
	cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
	clf = SVC(kernel='linear', random_state=241)
	gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
	gs.fit(X, y)
	for a in gs.grid_scores_:
		print a.mean_validation_score
		print a.parameters
	coef = gs.best_estimator_.coef_
	coef_ = coef.toarray()[0]
	top10 = np.argsort(abs(coef_))[-10:]
	feature_mapping = vzer.get_feature_names()
	for i in top10:
		print feature_mapping[i]



def step(w,x,y,c):
	w1,w2 = 0,0
	for i in range(len(y)):
		w1 += y[i]*x[i][0]*(1 - 1/(1+ np.exp(-y[i]*(w[0]*x[i][0] + w[1]*x[i][1]))))
		w2 += y[i]*x[i][1]*(1 - 1/(1+ np.exp(-y[i]*(w[0]*x[i][0] + w[1]*x[i][1]))))
	return np.array([w[0] + 0.1*w1/len(y) - 0.1*c*w[0],w[1] + 0.1*w2/len(y) - 0.1*c*w[1]])

def sigm(w,x):
	return 1/(1+np.exp(-w[0]*x[0] - w[1]*x[1]))

def as1():
	data = pandas.read_csv('svm-data.csv', header = None)
	X, y = data.ix[:,1:].as_matrix(), map(int,data.ix[:,0].as_matrix())
	clf = SVC(C = 100000,random_state = 241, kernel = 'linear')
	clf.fit(X,y)
	print clf.support_, clf.support_vectors_, X
def as3():
	data = pandas.read_csv('data-logistic.csv', header = None)
	x = data.ix[:,1:].as_matrix()
	y = map(lambda x:-1 if (x == 0) else x, map(int,data.ix[:,0].as_matrix()))
	w = np.array([0.,0.])
	for i in range(10000):
		w1 = step(w,x,y,0)
		if np.linalg.norm(w1-w) < 1e-5:
			print i
			break
		w = w1 
	#print w1
	w = np.array([0.,0.])
	for i in range(10000):
		w2 = step(w,x,y,10.)
		if np.linalg.norm(w2-w) < 1e-5:
			print i
			break
		w = w2 
	#print w2
	y_true = map(lambda x:0 if (x == -1) else x, map(int,data.ix[:,0].as_matrix()))
	#print y_true
	prob = map(lambda z: sigm(w1,z),x)
	#print prob
	print roc_auc_score(y_true,prob)
	prob = map(lambda z: sigm(w2,z),x)
	#print prob
	print roc_auc_score(y_true,prob)

def as4():
	data = pandas.read_csv('classification.csv')
	true,pred = data.ix[:,0].as_matrix(),data.ix[:,1].as_matrix()
	res = {"TP":0,'TN':0,'FP':0,'FN':0}
	for i in range(len(true)):
		if true[i] == 1:
			if pred[i] == 1:
				res['TP'] += 1
			else: res['FN'] += 1
		else:
			if pred[i] == 1:
				res['FP'] += 1
			else: res['TN'] += 1
	print metrics.accuracy_score(true,pred), metrics.precision_score(true,pred), metrics.recall_score(true,pred), metrics.f1_score(true,pred) 

def arrmax(x):
	max = 0
	prec = x[0]
	rec = x[1]
	for i in range(len(prec)):
		if rec[i] >= 0.7:
			if prec[i] > max: max = prec[i]
	return max

def as5():
	data = pandas.read_csv('scores.csv')
	true = data.ix[:,0].as_matrix()
	logreg = data.ix[:,1].as_matrix()
	svm	= data.ix[:,2].as_matrix()
	knn = data.ix[:,3].as_matrix()
	tree = data.ix[:,4].as_matrix()
	print arrmax(metrics.precision_recall_curve(true,logreg))
	print arrmax(metrics.precision_recall_curve(true,svm))
	print arrmax(metrics.precision_recall_curve(true,knn))
	print arrmax(metrics.precision_recall_curve(true,tree))
	#print roc_auc_score(true,logreg), roc_auc_score(true,svm), roc_auc_score(true,knn), roc_auc_score(true,tree)
as5()