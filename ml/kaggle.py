import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold,cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import time
import datetime 
df = pandas.read_csv('./features.csv', index_col='match_id')
df_test = pandas.read_csv('./features_test.csv', index_col='match_id')
heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero','r5_hero', 'd1_hero', 'd2_hero', 'd3_hero','d4_hero', 'd5_hero']
df.drop(['duration','tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'],inplace=True, axis = 1)
missing = []
df.fillna(0,inplace=True)
df_test.fillna(0,inplace=True)
l = len(df)
for col in df:
	if df[col].count() < l: missing.append(col)
y = df['radiant_win'].as_matrix()
df.drop(['radiant_win'],inplace=True,axis = 1)
X = df.as_matrix()
X_test = df_test.as_matrix()
def gradBoost():
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

def get_pairs():
	res = {}
	k = 0
	for i in range(1,113):
		for j in range(i+1,113):
			res[(i,j)] = k
			k += 1
	return res 

def prepData(X,df, train):
	scale = StandardScaler()
    #c = 0.001, 0.7162, 0.7163, 0.7517
	pairs = get_pairs()
	N = max(np.unique(df[heroes].values))
	print N
	X_pick = np.zeros((df.shape[0], N))
	X_pair = np.zeros((df.shape[0],N*(N-1)/2))
	print N*(N-1)/2, len(pairs)																																																				
	for i, match_id in enumerate(df.index):
		for p in xrange(5):
			X_pick[i, df.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
			X_pick[i, df.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
		for j in range(5):
			for k in range(j+1,5):
				tmp1 = tuple(sorted([df.ix[match_id, 'r%d_hero' % (j+1)],df.ix[match_id, 'r%d_hero' % (k+1)]]))
				tmp2 = tuple(sorted([df.ix[match_id, 'd%d_hero' % (j+1)],df.ix[match_id, 'd%d_hero' % (k+1)]]))
				X_pair[i,pairs[tmp1]] = 1
				X_pair[i,pairs[tmp2]] = -1

	df.drop(['lobby_type'] + heroes,inplace=True,axis=1)
	print X_pair.shape
	for i in range(X_pick.shape[1]):
		df[str(i) +' hero'] = X_pick[:,i]
	if train:
		for i in range(X_pair .shape[1]):
			if True:
				df[str(i) +' pair'] = X_pair[:,i]
				print i
			else:
				missing.append(i)
	else:
		for i in range(X_pair.shape[1]):
			if not (i in missing):
				print i
				df[str(i) +' pair'] = X_pair[:,i]		
	#y = df['radiant_win'].as_matrix()																													
	#X_pick = df.drop(['radiant_win'],axis = 1).as_matrix()
	print 123
	X = scale.fit_transform(sparse.csr_matrix(df.values.T))																																																																																																	
	return X

def logReg(X_test,X_train):
	clf = LogisticRegression(penalty='l2',C=0.01)
	clf.fit(X_train,y)
	res = clf.predict_proba(X_test)[:, 1]
	print res, max(res), min(res) #[ 0.64240285  0.71074219  0.59549928 ...,  0.46784135  0.4141525 0.91963519] 0.997600918121 0.00166936790687
	return res

def logRegC(X):

	cv = KFold(y.size, n_folds=5, shuffle=True)
	clf = LogisticRegression(penalty='l2',C=0.001)
	res = []
	for train,test in cv:
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		clf.fit(X_train,y_train)
		pred = clf.predict_proba(X_test)[:, 1]
		res.append(roc_auc_score(y_test, pred))
	res = np.array(res)
	print res, res.mean()
#gradBoost()
scale = StandardScaler()
X = scale.fit_transform(df.drop(['lobby_type'] + heroes,axis=1).as_matrix())
#logRegC(X)
skipped = []
X = prepData(X,df,True)
X_test = prepData(X_test,df_test,False)
r = logReg(X_test,X)
val = df_test.index.values
print len(r), len(val)
res = {'match_id': df_test.index.values,
		'radiant_win': r}
res = pandas.DataFrame(res)
print res
res.to_csv('res.csv',index=False)
