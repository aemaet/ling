import xml.etree.ElementTree as etree 
import pickle
import pymorphy2
import pandas
import time
import random
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
def read_data():
	data = pandas.read_csv('rt.csv')
	res = {}
	name1 = data['word1']
	name2 = data['word2']
	sim = data['sim']
	for i in range(len(name1)):
		if not (name1[i] in res): res[name1[i]] = set()
		else: 
			if sim[i] == 1: 
				res[name1[i]].add(name2[i])
	f = open('dict','wb')
	pickle.dump(res,f)
def clean(s):
	for i in range(len(s)):
		for sym in redundant: s[i] = s[i].replace(sym,'')
	return s

def norm(text):
	for i in range(len(text)):
		text[i] = morph.parse(text[i])[0].normal_form
	return text

def prep(ll,c):
	res = []
	for l in ll:
		l = norm(clean(l))
		while c in l:
			l.remove(c)
			res.append(l)
	return res

def prepData():

	out = open('norm','wb')
	res = []

	tree = etree.parse('text.xml')
	root = tree.getroot()
	for text in root:
		for paras in text.findall('paragraphs'):
			for para in paras.findall('paragraph'):
				for sent in para.findall('sentence'):
					res.append(norm(clean(sent.find("source").text.split())))
	
	pickle.dump(res,out)

def prepText():
	f = open("wiki.txt","r")
	out = open('wiki','wb')
	res = []
	for line in f:
		res+=prep([x.split() for x in (" ".join(line.split()[1:]).split('.'))],'')
	pickle.dump(res,out)

def split(a):
	i = a.index('$')
	return a[:i],a[i+1:]

def intersect(a,b,flag):
	l = min(len(a),len(b))
	res = ['$']
	if flag:
		for i in range(l):
			if a[i] == b[i]: res.append(a[i])
			else: break
	else:
		for i in reversed(range(l)):
			if a[i] == b[i]: res.append(a[i])
			else: break
		res = list(reversed(res))
	return res

def check_hyp(hyp):
	if hyp == ['$']: return False
	for w in hyp:
		if w == '$': continue
		if len(w) > 3: return True
		#p = morph.parse(w)[0]
		#if p.tag.POS == 'NOUN' or p.tag.POS == 'VERB': return True
	#if not res: print(hyp)
	return False


def build_hyp(facts):
	res = []
	start_time = time.time()
	print(len(facts))
	for i in range(len(facts)):
		a_l,a_r = split(facts[i])
		for j in range(i+1,len(facts)):
			b_l,b_r = split(facts[j])
			r1,r2 = intersect(a_l,b_l,False), intersect(a_r,b_r,True)
			r = r1+r2[1:]
			if check_hyp(r): res.append(r) 
	print(len(res))
	print("--- %s seconds build---" % (time.time() - start_time))
	return res 

def compare(h,s,t,f):
	#print(i,'/',t)
#	if word in s: return '' 
	pret = []
	res=''
	#hs = " ".join(h)
	ss = " ".join(s)
	if len(h.split('$')) != 2: return ''
	left,right = h.split('$')
	ls = left[:-1]
	rs = right[1:]
	if (not ls in ss) or (not rs in ss): return res
	for w in s:
		if not w in h: pret.append(w)
	p = ''
	for p in pret:
		#print(h.replace('$',p),ss)
		tmp = h.replace('$',p)
		if tmp in ss:
			if all([x in ss.split() for x in tmp.split()]):
				res = p
				break
	if res!='':
		if morph.parse(res)[0].tag.POS == 'NOUN': 
			#f.write(res + "! " + ss + "|" + h + "...\n")
			return res
	return ''


def hyp_dict(hyp):
	res = {}
	for h in hyp:
		h = " ".join(h)
		if h in res:
			res[h]+=1
		else:
			res[h]=1
	return res

def make_fact(s,w):
	res = []
	l = -1
	for i in range(len(s)):
		if s[i] == w and l == -1:
			l = 0
			continue
		if s[i] == w: 
			res.append(s[l:i])
			l = i
	res.append(s[l:])
	for i in range(len(res)):
		res[i] = ['$' if x==w else x for x in res[i]]
	return res

def analogy(hyp,text,index,ethalon):
	res = set()
	hyp_set = {}
	global i 
	f = open('res.txt','w')
	t = len(text)*len(hyp)
	#vec = []
	start_time = time.time()
	for h in hyp:
		h_list = h.split()
		if len(h_list) < 3: continue
		#hvec = hyp_to_vec(h_list,hyp[h])
		if h_list[0] != '$': pos = index[h_list[0]]
		else: pos = index[h_list[1]]
		for w in h_list:
			if w == '$':
				continue
			#print(len(pos))
			pos = pos.intersection(index[w])
		#print(h_list)
		#input(i)
		for s in pos:
		#	i+=1
			tmp = compare(h,text[s],t,f)	
			if tmp != '':
				if not h in hyp_set: hyp_set[h] = [tmp]
				else: hyp_set[h].append(tmp)  
				res.add(tmp)
		#if h in hyp_set: 
		#	hvec.append(eval_hyp(hyp_set[h],ethalon))
		#	vec.append(hvec)
	print("--- %s seconds compare---" % (time.time() - start_time))
	return res, hyp_set

def test_analogy(hyp,text,index,ethalon):
	res = set()
	hyp_set = {}
	global i 
	f = open('res.txt','w')
	t = len(text)*len(hyp)
	start_time = time.time()
	for h in hyp:
		h_list = h.split()
		if len(h_list) < 3: continue
		if h_list[0] != '$': pos = index[h_list[0]]
		else: pos = index[h_list[1]]
		for w in h_list:
			if w == '$':
				continue
			#print(len(pos))
			pos = pos.intersection(index[w])
		#print(h_list)
		#input(i)
		for s in pos:
		#	i+=1
			tmp = compare(h,text[s],t,f)	
			if tmp != '':
				if not h in hyp_set: hyp_set[h] = [tmp]
				else: hyp_set[h].append(tmp)  
				res.add(tmp)
	print("--- %s seconds compare---" % (time.time() - start_time))
	return res, hyp_set

def eval_hyp(s,es):
	s = set(s)
	#print(s,es)
	tmp = es.intersection(s)
	#print(len(s),len(tmp))
	if len(s) >= min(len(tmp) + 40, len(tmp)*10): res = 0
	else: res = 1
	#print(res)
	return res

def hyp_to_vec(h,amount):
	#print(h)
	res = [0.]*16
	l = len(h)
	res[0] = amount
	for w in h:
		if w =='$': continue
		p = morph.parse(w)[0].tag.POS
		#print(w,p)
		if p == 'NOUN': res[1] += 1
		elif p == 'ADJF': res[2] += 1
		elif p == 'ADJS': res[3] += 1
		elif p == 'COMP': res[4] += 1
		elif p == 'INFN': res[5] += 1
		elif p == 'PRTF': res[6] += 1
		elif p == 'PRTS': res[7] += 1
		elif p == 'GRND': res[8] += 1
		elif p == 'NUMR': res[9] += 1
		elif p == 'ADVB': res[10] += 1
		elif p == 'PRED': res[11] += 1
		elif p == 'PREP': res[12] += 1
		elif p == 'CONJ': res[13] += 1
		elif p == 'PRCL': res[14] += 1
		elif p == 'INTJ': res[15] += 1
	res = [res[0]] + [x/l for x in res[1:]] + [l]
	return res


def purge_set(s, index):
	res = set()
	for i in s:
		if i in index: res.add(i)
	return res

def main():
	text_t = open('wiki','rb')
	index_t = open('index','rb')	
	text = pickle.load(text_t)
	index = pickle.load(index_t)
	out = open('output.txt','w')
	factsf = open('facts.txt', 'w')
	d_t = open('dict','rb')
	out_hyp = open('out.txt','w')
	d = pickle.load(d_t)
	#inp = random.sample(list(d),100)#['гора']
	#inp = ['конфликт']
	recall = []
	i = 1
	X = []
	#print(inp)
	for word in ['гора']:
		try:
			if len(recall) >= 10: break
			word = morph.parse(word)[0].normal_form
			print(len(index[word]), ' enterance')
			if len(index[word]) > 5000 or len(index[word]) < 500: continue
			true_res = purge_set(d[word],index)
			if len(true_res) == 0: continue
			facts = []
			hyp = []
			print(word)
			for i in index[word]:
				tmp = make_fact(text[i],word)
				factsf.write(" ".join(tmp[0]))
				for t in tmp:
					facts.append(t)
			hyp+=build_hyp(facts)
			hyp = hyp_dict(hyp)
			res, hres= analogy(hyp,text,index,true_res)
			if len(true_res.intersection(res)) == 0: continue
			start_time = time.time()
			print(true_res,true_res.intersection(res),len(res))
			recall.append(float(len(true_res.intersection(res)))/len(true_res))
			print(len(recall), ' processed')
			res = set()
			#hyp = [" ".join(x) for x in hyp]	
			for h in hres:
				y = eval_hyp(hres[h],true_res)
				X.append(hyp_to_vec(h.split(),hyp[h]) + [y])
				if y == 1:
					res.update(hres[h]) 
					out_hyp.write(h + ' : ' + " ".join(set(hres[h])) + '\n')
			print(true_res,true_res.intersection(res),len(res))
			out.write("\n".join(hyp))
			print("--- %s seconds vector---" % (time.time() - start_time))
		except KeyError: 
			print('snap')
			recall.append(1.0)
	res= pandas.DataFrame.from_records(np.array(X))
	print(res.shape)
	res.to_csv('res_copy.csv',index=False)
	print(sum(recall)/len([x for x in recall if x > 0]))


def reverse_index(t): 
	out = open('index','wb')
	res = {}
	print(len(t))
	for i in range(len(t)):
		for w in t[i]:
			if w in res: res[w].add(i)
			else: res[w] = {i}
	print(res['кот'])
	pickle.dump(res,out)

def train_rfc():
	out = open('rfc','wb')
	data = pandas.read_csv('res_copy.csv')
	X = data.ix[:,1:-2]
	y = data.ix[:,-1]
	#scale = StandardScaler()
	#X = scale.fit_transform(X)
	clf = RandomForestClassifier()
	#clf.fit(X,y)
	#pickle.dump(clf,out)
	#grid = {'C': np.power(10.0, np.arange(-5, 6))}
	#gs = grid_search.GridSearchCV(clf, grid)
	res = cross_validation.cross_val_score(clf, X, y, cv=5,scoring='roc_auc')
	print(res, res.mean())
	clf = GradientBoostingClassifier()
	clf.fit(X,y)
	pickle.dump(clf,out)
	res = cross_validation.cross_val_score(clf, X, y, cv=5,scoring='roc_auc')
	clf.fit(X,y)
	print(res, res.mean(),clf.feature_importances_)

def train():
	out = open('logreg','wb')
	data = pandas.read_csv('res_copy.csv')
	X = data.ix[:,1:-1]
	y = data.ix[:,-1]
	scale = StandardScaler()
	X = scale.fit_transform(X)
	cv = KFold(y.size, n_folds=5, shuffle=True)
	clf = LogisticRegression(penalty='l2', C=0.1)
	#res = cross_validation.cross_val_score(clf, X, y, cv=5,scoring='roc_auc')
	#clf.fit(X,y)
	#pickle.dump(clf,out)
	grid = {'C': np.power(10.0, np.arange(-5, 6))}
	gs = grid_search.GridSearchCV(clf, grid)
	res = []
	for train,test in cv:
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		gs.fit(X_train,y_train)
		pred = gs.predict_proba(X_test)[:, 1]
		print (gs.best_estimator_.C)
		#pred = clf.predict_proba(X)[:, 1]
		res.append(roc_auc_score(y_test, pred))
	res = np.array(res)
	print(res, res.mean())

def clear_hyp(hyp,clf):
	res = []
	hyp_list = []
	vec_list = []
	for h in hyp:
		hyp_list.append(h)
		vec_list.append(np.array(hyp_to_vec(h.split(),hyp[h])))
	pred = clf.predict(vec_list)
	print(np.sum(pred))
	for i in range(len(hyp_list)):
		if pred[i] == 1: res.append(hyp_list[i]) 
	return res

def fmeasure(res,true):
	tmp = float(len(true.intersection(res)))
	if tmp == 0: return [1,1,1]
	prec = tmp/len(res)
	recall = tmp/len(true)
	f = 2*prec*recall/(prec+recall)
	print(prec,recall,f)
	return [prec,recall,f]

	

def test():
	text_t = open('wiki','rb')
	index_t = open('index','rb')	
	text = pickle.load(text_t)
	index = pickle.load(index_t)
	out = open('output.txt','w')
	d_t = open('dict','rb')
	out_hyp = open('out.txt','w')
	clf_t = open('rfc','rb')
	d = pickle.load(d_t)

	data = pandas.read_csv('res_copy.csv')
	X = data.ix[:,:-1]
	y = data.ix[:,-1]
	#clf = pickle.load(clf_t)
	clf = GradientBoostingClassifier()
	clf.fit(X,y)
	#inp = random.sample(list(d),10)#['гора']
	inp = ['конфликт','трагедия','гора']
	#inp=['распад','сеть', 'завершение']
	#inp = ['руководство']
	fm = []
	print(inp)
	for word in inp:
		try:
			word = morph.parse(word)[0].normal_form
			true_res = purge_set(d[word],index)
			facts = []
			hyp = []
			print(word)
			for i in index[word]:
				tmp = make_fact(text[i],word)
				for t in tmp:
					facts.append(t)
			hyp+=build_hyp(facts)
			print(len(hyp))
			
			
			hyp = hyp_dict(hyp)
			hyp = clear_hyp(hyp,clf)
			print(len(hyp))
			res, hres= test_analogy(hyp,text,index,true_res)
			if len(res) == 0: continue
			print(len(fm))
			print(true_res,true_res.intersection(res),len(res))
			#print(recall)
			#res = set()
			#hyp = [" ".join(x) for x in hyp]	
			#for h in hres:
			#	y = eval_hyp(hres[h],true_res)
			#	if y == 1:
			#		res.update(hres[h]) 
			#		out_hyp.write(h + ' : ' + " ".join(set(hres[h])) + '\n')
			print(true_res,true_res.intersection(res),len(res))
			if len(true_res) == 0 or len(res) == 0: fm.append([1,1,1])
			else: fm.append(fmeasure(res,true_res))
			out.write("\n".join(hyp))
		except KeyError: continue
	fm = np.array(fm)
	print(np.mean(fm,axis=0))
	#print(sum(recall)/len([x for x in recall if x > 0]))

#global i
i = 0
morph = pymorphy2.MorphAnalyzer()
word = 'дождь'
redundant = [',', '-', '—', '.', '…', '!', ':', '?', '»', '«', '(', ')', '\n']
#reverse_index(text)
#read_data()
main()
#test()
#train_rfc()
#train()
#f = open('wiki.txt','r')
#line = f.readline()
#print([x.split() for x in (" ".join(line.split()[1:]).split('.'))])