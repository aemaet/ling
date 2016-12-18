import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cosine 
from sklearn.metrics import pairwise_distances


def matc(x):
	res = []
	for i in x:
		for j in x:
			res.append(cosine(np.array(i),np.array(j))*100)
	return np.array(res)
def mat(x):
	res = []
	for i in x:
		for j in x:
			res.append(np.linalg.norm(np.array(i)-np.array(j)))
	return np.array(res)

a = [[0.6,1.9],[1.8,1.6],[2.7,2.0],[3.0,2.1],[3.0,2.6],[3.1,4.5],[3.8,0.6],[4.2,4.7]]
b = mat(a).reshape(len(a),len(a))
c = np.array_repr(matc(a).reshape(len(a),len(a)),precision=4,suppress_small=True)
print b
print c