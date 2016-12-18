# -*- coding: latin-1 -*-
import scipy.integrate as integ
import random
class Rule:
	cond_prob = 0.
	def __init__(self,cond,conc):
		self.condition = cond
		self.conclusion = conc
	def activate(self,x):
		return min([self.cond_prob,self.conclusion.member(x)])

def membership_h(x,a,b):
	if x <= a: return 0
	if a < x <= b: return (x-a)/(b-a)
	else: return 1

def membership_m(x,a,b,c):
	if x <= a: return 0
	if a < x <= b: return (x-a)/(b-a)
	if b < x <= c: return (c-x)/(c-b)
	else: return 0

def membership_l(x,a,b):
	if x <= a: return 1
	if a < x <= b: return (b-x)/(b-a)
	else: return 0

class Score:
	a_l = 10
	b_l = 30
	a_lm = 20
	b_lm = 30
	c_lm = 40
	a_m = 30
	b_m = 50
	c_m = 70
	a_hm = 60
	b_hm = 75
	c_hm = 90
	a_h = 70
	b_h = 90
	def __init__(self, label):
		self.label = label
	def score(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "lmed": return membership_m(x,self.a_lm,self.b_lm,self.c_lm)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "hmed": return membership_m(x,self.a_hm,self.b_hm,self.c_hm)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Time:
	a_l = 60
	b_l = 90
	a_m = 60
	b_m = 120
	c_m = 180
	a_h = 120
	b_h = 180
	def __init__(self, label):
		self.label = label
	def time(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)	

class Health:
	a_l = 25
	b_l = 50
	a_m = 25
	b_m = 50
	c_m = 75
	a_h = 50
	b_h = 75
	def __init__(self, label):
		self.label = label
	def health(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Eng_gram:
	name = "eng_gr"
	a_l = 0
	b_l = 4
	a_m = 2
	b_m = 5
	c_m = 8
	a_h = 7
	b_h = 10
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Fr_gram:
	name = "fr_gr"
	a_l = 0
	b_l = 4
	a_m = 2
	b_m = 5
	c_m = 8
	a_h = 7
	b_h = 10
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Length:
	name = "length"
	a_l = 2
	b_l = 4
	a_m = 3
	b_m = 5
	c_m = 7
	a_h = 6
	b_h = 8
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Special_sym:
	name = "special_sym"
	a_l = 0
	b_l = 20
	a_m = 10
	b_m = 200
	c_m = 300
	a_h = 250
	b_h = 350
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)


class Su_gram:
	name = "su_gr"
	a_l = 100
	b_l = 200
	a_m = 150
	b_m = 300
	c_m = 450
	a_h = 400
	b_h = 500
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Sp_gram:
	name = "sp_gr"
	a_l = 30
	b_l = 50
	a_m = 40
	b_m = 80
	c_m = 120
	a_h = 100
	b_h = 150
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Pl_gram:
	name = "pl_gr"
	a_l = 100
	b_l = 200
	a_m = 150
	b_m = 300
	c_m = 450
	a_h = 400
	b_h = 500
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "low": return membership_l(x,self.a_l,self.b_l)
		if self.label == "med": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "high": return membership_h(x,self.a_h,self.b_h)

class Language:
	a_l = 10
	b_l = 30
	a_lm = 20
	b_lm = 30
	c_lm = 40
	a_m = 30
	b_m = 50
	c_m = 70
	a_hm = 60
	b_hm = 75
	c_hm = 90
	a_h = 70
	b_h = 90
	def __init__(self, label):
		self.label = label
	def member(self,x):
		if self.label == "english": return membership_l(x,self.a_l,self.b_l)
		if self.label == "french": return membership_m(x,self.a_lm,self.b_lm,self.c_lm)
		if self.label == "finnish": return membership_m(x,self.a_m,self.b_m,self.c_m)
		if self.label == "polish": return membership_m(x,self.a_hm,self.b_hm,self.c_hm)
		if self.label == "spanish": return membership_h(x,self.a_h,self.b_h)

def lexim(x):
	if x <= 20: return "English"
	if x <= 40: return "French"
	if x <= 60: return "Finnish"
	if x <= 80: return "Polish"
	else: return "Spanish"

def defuzz(x):
	global union
	return max([f(x) for f in union])

rules = [Rule([Special_sym("low"),Eng_gram("high"),Length("med")],Language("english")),
		Rule([Special_sym("med"),Fr_gram("high"),Length("med")],Language("french")),
		Rule([Special_sym("high"),Su_gram("high"),Length("high")],Language("finnish")),
		Rule([Special_sym("low"),Pl_gram("high"),Length("high")],Language("polish")),
		Rule([Special_sym("low"),Sp_gram("high"),Length("med")],Language("spanish"))]

features = {'eng_gr':0,
			'fr_gr':0,
			'special_sym':0,
			'su_gr':0,
			'pl_gr':0,
			'sp_gr':0,
			'length':0}
inputf = open("text_sp.txt","r")
eng_gr =[" do "," did ", " was ", "were", " is ", " are ", " am "]
fr_gr = [" que ","oix","aux"]
pl_gr = ["ż","z","w","ś"]
sp_gr = [" la ", " lo ", " de ", " se ", "ó", "á"]
s_sym = ["ù","û","ü","ÿ","à","â","ä","æ","ç","é","è","ê","ë","ï","î","ô","œ"]
length = 0
temp = None
for line in inputf:
	for gram in eng_gr:
		#print sym
		if gram in line: features['eng_gr'] += line.count(gram)
	for gram in fr_gr:
		#print sym
		if gram in line: features['fr_gr'] += line.count(gram)
	for gram in pl_gr:
		#print sym
		if gram in line: features['pl_gr'] += line.count(gram)
	for gram in sp_gr:
		#print sym
		if gram in line: features['sp_gr'] += line.count(gram)
	for ch in line:
		if temp == ch: features['su_gr']+=1
		temp = ch
	for sym in s_sym:
		#print sym
		if sym in line: features['special_sym'] += line.count(sym)
	#print line
	features['length']+=len(line)
	length += len(line.split())
for key in features.keys():
	if key == "length": features[key] /= length
	else:features[key] = 1000*features[key]/length
print features
union = []
for rule in rules:
	print [x.member(float(features[x.name])) for x in rule.condition]
	rule.cond_prob = min([x.member(float(features[x.name])) for x in rule.condition])#[rule.condition[0].time(i[0]), rule.condition[1].health(i[1])])
	union.append(rule.activate)
	#print membership_m(300.,60,120,180)
res = integ.quad(lambda x: x*defuzz(x),0.,100.)[0]/integ.quad(lambda x: defuzz(x),0.,100.)[0]
print res
print lexim(res)