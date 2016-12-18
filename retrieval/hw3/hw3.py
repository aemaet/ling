import pymorphy2
import math
from operator import itemgetter
morph = pymorphy2.MorphAnalyzer()

def normal_query(q):
	res = []
	for i in range(len(q)):
		res.append(morph.parse(q[i])[0].normal_form)
	return res

def tf(w, s):
	res = 0
	for i in range(len(s)):
		if (w == s[i]): res += 1
	return res/len(s)

def idf(w):
	res = 0
	for s in text:
		if (w in s): res +=1
	if (res == 0): return 0
	return math.log(len(text)/res) 

def weight(words, s):
	res = []
	for i in range(len(words)):
		res.append(tf(words[i], s) * idf(words[i]))
	return res

def dot(v1, v2):
	res = 0
	for i in range(len(v1)):
		res+= v1[i]*v2[i]
	return res 	

def sim(v1,v2):
	if (dot(v2,v2) == 0): return 0
	return dot(v1,v2)/(math.sqrt(dot(v1,v1)) * math.sqrt(dot(v2,v2)))

def ndcg(rank):
	res1 = rank[0]
	for i in range(1,len(rank)):
		res1 += rank[i]/(math.log(i+1,2))
	rank.sort(reverse = True)
	res2 = rank[0]
	for i in range(1,len(rank)):
		res2 += rank[i]/(math.log(i+1,2))
	return res1/res2 

docs = open('docs', 'r')
out = open('outputidf.txt', 'w')
text = []
redundant = [',', '–', '—', '.', '…', '!', ':', '?', '»', '«', '(', ')']
for line in docs:
	for sym in redundant: line = line.replace(sym,'')
	text.append(normal_query(line.split()))
q = [None]*3
qset = [None]*3
qv= [None]*3
q[0] = "Контактёрами с инопланетянами в своё время называли себя и польский эмигрант и президент Калмыкии".split()
q[1] = "Колумба и его команду от голодной смерти спасло лунное затмение".split()
q[2] = "Скатологическая детская литература началась 26 лет назад со сказки про кротика".split()
for i in range(len(q)): q[i] = normal_query(q[i])
for i in range(len(q)): qset[i] = list(set(q[i]))
for i in range(len(q)): qv[i] = weight(qset[i], q[i])
res = []
for i in range(len(q)):
	for j in range(len(text)):
		res.append([text[j], sim(qv[i], weight(qset[i],text[j]))])

	res.sort(key=itemgetter(1), reverse = True)
	for item in res:
	    if (item[1] != 0): out.write(''.join([' '.join(item[0]), ' ', str(item[1]), '\n \n']))
	out.write('\n ================================================== \n \n')
	res = []

