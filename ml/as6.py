import numpy as np
from skimage.io import imread
import skimage
import pylab
from sklearn.cluster import KMeans

img = imread('parrots.jpg')
image = skimage.img_as_float(imread('parrots.jpg'))
print image
X = [x for y in image for x in y]
n = len(image)
m = len(image[0])
print n,m
#print X
km = KMeans(n_clusters=11,random_state=241)
k = km.fit_predict(X)
clust = {}
for i in range(len(X)):
	if k[i] in clust.keys():
		clust[k[i]] = np.vstack((clust[k[i]],X[i]))
	else:
		clust[k[i]] = np.array(X[i])
#print clust
mean,median = {},{}
for key in clust:
	mean[key] = np.mean(clust[key],0)
	median[key] = np.median(clust[key],0)
#print mean
img_mean, img_med = [],[]
for i in range(len(X)):
	img_mean.append(mean[k[i]])
	img_med.append(median[k[i]])
mse_mean = ((np.array(img_mean) - np.array(X))**2).mean() / 3 
mse_med = ((np.array(img_med) - np.array(X))**2).mean() / 3
res_mean = 10*np.log10((1.0)/mse_mean) 
res_med = 10*np.log10((1.0)/mse_med) 
print res_mean, res_med
print ((20 * np.log10(1)) - (10 * np.log10(mse_mean)))
pylab.imshow(img)