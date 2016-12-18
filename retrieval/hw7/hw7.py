import xml.etree.ElementTree as etree 
from itertools import chain
import pymorphy2
import math
def normal_q(q):
	res = []
	for i in range(len(q)):
		res.append(morph.parse(q[i])[0].normal_form)
	return res

def mi(q,n):
	fab = text_s.count(q)
	if fab <= n: return -1
	fa = text_s.count(q.split()[0])
	fb = text_s.count(q.split()[1])
	print(fab,fa,fb)
	return math.log(fab*len(text)/(fa*fb),2)

morph = pymorphy2.MorphAnalyzer()
inp = open('ling.txt','r')
text = []
redundant = [',', '–', '—', '.', '…', '!', ':', '?', '»', '«', '(', ')']
for line in inp:
	for sym in redundant: line = line.replace(sym,'')
	text.append(normal_q(line.split()))
text_s = ' '.join(list(chain.from_iterable(text)))
tree = etree.parse('output.xml')
root = tree.getroot()
col = []
#print(root[0][0][0][0].get('val'))
for fact in root[0][0]:
	temp = []
	temp.append(fact[0].get('val').lower())
	temp.append(fact[1].get('val').lower())
	col.append(' '.join(temp))
col = list(set(col))

res_freq = []
res_mi = []
res_bound = []

for i in range(len(col)):
	res_freq.append((text_s.count(col[i]),i))
	res_mi.append((mi(col[i],0),i))
	if mi(col[i],4) != -1: res_bound.append((mi(col[i],4),i))
res_freq.sort(reverse = True)
res_mi.sort(reverse = True)
res_bound.sort(reverse = True)

#print(len(res_freq),len(res_mi),len(res_bound))

out = open('result.txt','w')
out.write('ЧАСТОТНАЯ \n')
for i in range(20): 
	out.write(str((col[res_freq[i][1]],res_freq[i][0])))
	out.write('\n')
out.write('\n ================================================== \n ОБЩАЯ\n')
for i in range(20): 
	out.write(str((col[res_mi[i][1]],res_mi[i][0])))
	out.write('\n')
out.write('\n ================================================== \n ОГРАНИЧЕННАЯ\n')
for i in range(20): 
	out.write(str((col[res_bound[i][1]],res_bound[i][0])))
	out.write('\n')
out.write('\n ================================================== \n \n')
	#print(col[res_bound[i][1]],res_bound[i][0])
