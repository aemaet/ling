#coding=utf8
import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
import time
import datetime 

#B
#Градиентный бустинг 
def gradBoost(X,y):
	cv = KFold(y.size, n_folds=5, shuffle=True)
	clf = GradientBoostingClassifier(n_estimators=30)
	res = []
	start_time = datetime.datetime.now()
	for train,test in cv:
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		clf.fit(X_train,y_train)
		pred = clf.predict_proba(X_test)[:, 1]
		res.append(roc_auc_score(y_test, pred))
	res = np.array(res)
	print res, res.mean()	
	print 'Time elapsed:', datetime.datetime.now() - start_time

#E 
#превращение данных о героях в мешок слов 
def prepData(X,df):
	scale = StandardScaler()
    #c = 0.001, 0.7162, 0.7163, 0.7517
    #D
	N = max(np.unique(df[heroes].values))
	X_pick = np.zeros((df.shape[0], N))
	for i, match_id in enumerate(df.index):
		for p in xrange(5):
			X_pick[i, df.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
			X_pick[i, df.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
	df.drop(['lobby_type'] + heroes,inplace=True,axis=1)
	for i in range(X_pick.shape[1]):
		df[str(i) +' hero'] = X_pick[:,i]
	#y = df['radiant_win'].as_matrix()
	#X_pick = df.drop(['radiant_win'],axis = 1).as_matrix()
	X = scale.fit_transform(df.as_matrix())
	return X

#F
#логистическая регрессия с оптимальными параметрами на новой выборке
def logReg(X_test,X_train):
	clf = LogisticRegression(penalty='l2',C=0.001)
	clf.fit(X_train,y)
	res = clf.predict_proba(X_test)[:, 1]
	print res, max(res), min(res) #[ 0.64240285  0.71074219  0.59549928 ...,  0.46784135  0.4141525 0.91963519] 0.997600918121 0.00166936790687
	return res
#C
#логистическая регрессия, подбор оптимального С
def logRegC(X,y):
	cv = KFold(y.size, n_folds=5, shuffle=True)
	clf = LogisticRegression(penalty='l2')
	grid = {'C': np.power(10.0, np.arange(-5, 6))}
	gs = grid_search.GridSearchCV(clf, grid)
	res = []
	for train,test in cv:
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		gs.fit(X_train,y_train)
		pred = gs.predict_proba(X_test)[:, 1]
		print gs.best_estimator_.C
		res.append(roc_auc_score(y_test, pred))
	res = np.array(res)
	print res, res.mean()


df = pandas.read_csv('./features.csv', index_col='match_id')
df_test = pandas.read_csv('./features_test.csv', index_col='match_id')
heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero','r5_hero', 'd1_hero', 'd2_hero', 'd3_hero','d4_hero', 'd5_hero']
#убирание признаков, заглядывающих в будущее
df.drop(['duration','tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'],inplace=True, axis = 1)

#A
#Нахождение пропусков
missing = []
l = len(df)
for col in df:
	if df[col].count() < l: missing.append(col)
print missing
#заполнение пропусков, формирование массива объектов
df.fillna(0,inplace=True)
df_test.fillna(0,inplace=True)
y = df['radiant_win'].as_matrix()
df.drop(['radiant_win'],inplace=True,axis = 1)
X = df.as_matrix()
X_test = df_test.as_matrix()

gradBoost(X,y)
#масштабирование признаков, убираются категориальные признаки
scale = StandardScaler()
X = scale.fit_transform(df.drop(['lobby_type'] + heroes,axis=1).as_matrix())
#превращение данных о героях в мешок слов
X = prepData(X,df)
X_test = prepData(X_test,df_test)
#r = logReg(X_test,X)

#Запись результатов в csv
#val = df_test.index.values
#res = {'match_id': df_test.index.values,
#		'radiant_win': r}
#res= pandas.DataFrame(res)
#print res
#res.to_csv('res.csv',index=False)
