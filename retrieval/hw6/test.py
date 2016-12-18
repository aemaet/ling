import xml.etree.ElementTree as etree    
from numpy import *
import pymorphy2

def clean(s):
	for i in range(len(s)):
		for sym in redundant: s[i] = s[i].replace(sym,'')
	return s

def norm(text):
	for i in range(len(text)):
		text[i] = morph.parse(text[i])[0].normal_form
	return text

def quantity(e, l):
	res = 0
	for n in classes[l]:
		res+=n.count(e)
	return res 

def len_sum(l):
	res = 0
	for n in l:
		res += len(n)
	return res

def prob(query,c):
	p = []
	lens = c_len[c]
	for i in range(len(query)):
		t_temp = quantity(query[i],c)  
		p.append((t_temp+1)/(lens+len(query)))
	return p

def classify(q):
	prob_q= [0,0,0]
	for i in range(len(prob_q)):
		prob_q[i] = log(prob_c[i]) + sum(log(prob(q,i)))
		if i == 0: prob_q[i]*=0.5
	#print(prob_q)
	return prob_q.index(max(prob_q))

def check(l1,l2,t):
	res=[0,0,0,0]#tp,fp,fn,tn
	for i in range(len(l1)):
		for j in range(i,len(l2)):
			if l1[i][0] != l2[j][0]: continue
			if l1[i][1] == t*1:
				if l2[j][1] == t*1: res[0]+=1
				else: res[2]+=1
			else:
				if l2[j][1] != t*1: res[3]+=1
				else: res[1]+=1
			break
	for i in range(len(res)):
		if res[i] == 0: res[i] = 1
	return res

morph = pymorphy2.MorphAnalyzer()
redundant = ['http://t.co/',',', '–', '.', '…', '!', ':', '?', '»', '«', '(', ')','@']
tree = etree.parse('bank_train.xml')
test_tree = etree.parse('bank_test.xml')   
root = tree.getroot()
test_root = test_tree.getroot()
banks = ["sberbank", "vtb", "gazprom", "alfabank", "bankmoskvy", "raiffeisen", "uralsib", "rshb"]  
classes = [[],[],[]]

etalon=[]
e_tree = etree.parse('etalon.xml')  
e_root = e_tree.getroot()
for tweet in e_root[1].findall('table'):
	ind = tweet.find('column[@name="id"]').text
	for bank in banks:
		val = tweet.find("column[@name='" + bank + "']").text
		if val != "NULL": etalon.append((int(ind),int(val)))

for tweet in root[1].findall('table'):
	for bank in banks:
		val = tweet.find("column[@name='" + bank + "']").text
		if 	val == "1": 
			classes[0].append(norm(clean(tweet.find("column[@name='text']").text.split())))
			break
		if val == "0": 
			classes[1].append(norm(clean(tweet.find("column[@name='text']").text.split())))
			break
		if val == "-1": 
			classes[2].append(norm(clean(tweet.find("column[@name='text']").text.split())))
			break
lsum = len(classes[0])+len(classes[1])+len(classes[2]) 

result = []

horror = len(test_root[1].findall('table'))
prec_p,prec_n,recall_p,recall_n = 0.25,0.25,0.25,0.25
c_len = [len_sum(classes[0]),len_sum(classes[1]),len_sum(classes[2])]
prob_c =[len(classes[0])/lsum,len(classes[1])/lsum,len(classes[2])/lsum]	
h_count = 0
print(lsum, prob_c[0],prob_c[1],prob_c[2])
for tweet in test_root[1].findall('table'):
	ind = tweet.find('column[@name="id"]').text
	for bank in banks:
		val = tweet.find("column[@name='" + bank + "']").text
		if val == "0":
			t = norm(clean(tweet.find("column[@name='text']").text.split()))
			temp = classify(t)
			if temp == 0: result.append((int(ind),1))
			elif temp == 1: result.append((int(ind),0))
			else: result.append((int(ind),-1))#tweet.find("column[@name='" + bank + "']").text = "-1" 
			break
	h_count+=1
	print(h_count)
	#if h_count > 10: break 

print(len(etalon),len(result))
fp = check(etalon,result,1)
fn = check(etalon,result,-1)
prec_p = fp[0]/(fp[0]+fp[1])
recall_p = fp[0]/(fp[0] + fp[2])
#print(prec, recall)
fp = 2*prec_p*recall_p/(prec_p+recall_p)
prec_n = fn[0]/(fn[0]+fn[1])
recall_n = fn[0]/(fn[0] + fn[2])
fn = 2*prec_n*recall_n/(prec_n+recall_n)
#print(prec, recall)
print(fp,fn,(fp+fn)/2)
#print(root[1].find('table').find("column[@name='sberbank']"))
#print("column[@name='" + banks[0] + "']")
#print(root[1].find('table').find("column[@name='" + banks[0] + "']"))
		