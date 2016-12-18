import numpy as np
n = 3
d = 0.1
mat = np.array([
	[d/n,(1-d)/2 + d/n,(1-d)/2 + d/n],
	[d/n,d/n,1-d + d/n],
	[d/n,1 - d + d/n,d/n]])
s = np.array([1.0/3,1.0/3,1.0/3])
t = np.dot(s,mat)
count = 0
while np.linalg.norm(s-t) > 0.0001:
	count += 1
	temp = t
	t = np.dot(t,mat)
	s = temp
print np.linalg.norm(s-t)
print np.linalg.eig(mat)
print mat
print t,count  