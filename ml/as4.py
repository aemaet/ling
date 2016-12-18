from sklearn.feature_extraction import DictVectorizer
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np
def as1():
	data_train = pandas.read_csv('salary-train.csv')
	data_test = pandas.read_csv('salary-test-mini.csv')
	data_train['FullDescription'] = data_train['FullDescription'].apply(lambda x: x.lower())
	data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
	data_test['FullDescription'] = data_test['FullDescription'].apply(lambda x: x.lower())
	data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
	#data_train['LocationNormalized'] = data_train['LocationNormalized'].apply(lambda x: x.lower())
	#data_train['LocationNormalized'] = data_train['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
	data_train['LocationNormalized'].fillna('nan', inplace=True)
	data_train['ContractTime'].fillna('nan', inplace=True)
	vzer = TfidfVectorizer(min_df=5)
	y = data_train['SalaryNormalized'].as_matrix()
	#print y
	#print data_train.ix[:,0]
	X = vzer.fit_transform(data_train.ix[:,0])
	Y = vzer.transform(data_test.ix[:,0])
	enc = DictVectorizer()
	X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
	#print X.toarray()
	#print len(X),len(X[0]),len(X_train_categ),len(X_train_categ[0]) 
	X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
	#print X_test_categ
	x_train = hstack([X,X_train_categ])
	x_test = hstack([Y,X_test_categ])
	clf = Ridge(alpha=1, random_state=241)
	#X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
	#print X_test_categ
	clf.fit(x_train,y)
	print clf.predict(x_test)

data_train = pandas.read_csv('close_prices.csv')
X = data_train.ix[:,1:].as_matrix()
pca = PCA(n_components=10)
pca.fit(X)
#print pca.explained_variance_ratio_
tmp = pca.transform(X)[:,0]
#print len(tmp), len(tmp[0])
data = pandas.read_csv('djia_index.csv')
y = data.ix[:,1].as_matrix()
print tmp, y
print np.corrcoef(tmp,y)
m = max(abs(tmp))
print abs(tmp).argmax(), len(tmp)
print data_train.columns[np.argmax(pca.components_[0])+1]