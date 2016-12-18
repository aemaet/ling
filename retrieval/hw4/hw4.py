import math
from numpy import * 
qDocs = 1000000
qSelection = 70000
qCandidate = 70000
qSelect = 50000
qApplicant = 30000
qChoose = 70000 
a = 0.7
b = 0.3
iniQuery = array([1,0,1,0,0]) #selection, select, candidate, choose, applicant
revDoc1 = linalg.norm(array([0,log10(qDocs/qCandidate),log10(qDocs/qSelect),0,log10(qDocs/qApplicant)]))
revDoc2 = linalg.norm(array([log10(qDocs/qSelection),0,0,log10(qDocs/qChoose), log10(qDocs/qApplicant)]))

res = a*iniQuery + b*(revDoc1 + revDoc2)
print (res)

#[ 1.04647059  0.34647059  1.090309    0.34647059  0.91372725]