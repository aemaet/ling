from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas
from sklearn.feature_extraction import DictVectorizer as DV
import numpy as np


def draw():
	df = pandas.read_csv('res_copy (copy).csv').fillna('nan')
	#print df.shape
	#vectorizer = DV( sparse = False )
	#df_dict = df.drop( [ '0', '11', '12'], axis = 1 ).T.to_dict().values()
	#df_dict = df_dict.fillna( 'NA' )
	#print df_dict
	#X = vectorizer.fit_transform(df_dict)
	#print X.shape
	X = df.ix[:,1:-2].as_matrix()
	y = df.ix[:,-1].as_matrix()
	#scale = StandardScaler()
	#X = scale.fit_transform(X)
	print len([x for x in y if x == 0])
	pca = PCA(n_components=2)
	
	pca.fit_transform(X)
	#print(pca.components_)
	print(pca.explained_variance_ratio_) 
	print len(X[y == 1, 0]), len(X[y == 1, 1])
	plt.figure()
	for c, i, target_name in zip("kw", [1, 0], ['correct','incorrect']):
		plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
	plt.xlim([-10,100])
	plt.ylim([-50,100])
	plt.legend()
	plt.title('PCA')
	plt.show()

draw()