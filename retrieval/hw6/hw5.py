from numpy import *
import pymorphy2
from colorama import Fore, Style

def norm(text):
	for i in range(len(text)):
		for j in range(len(text[i])):
			text[i][j] = morph.parse(text[i][j])[0].normal_form
	return text

def quantity(e, l):
	res = 0
	for n in l:
		if isinstance(n, list): res+= quantity(e,n) 
		if l.count(e) > 0: 
			res+=l.count(e)
			break
	return res 

def len_sum(l):
	res = 0
	for n in l:
		res += len(n)
	return res

def prob(query,c):
	p = []
	lens = len_sum(c)
	for i in range(len(query)):
		t_temp = quantity(query[i],c) 
		p.append((t_temp+1)/(lens+len(query)))
	return p

def classify(q):
	prob_a = log(pc1) + sum(log(prob(q,classes[0])))
	prob_b = log(pc2) + sum(log(prob(q,classes[1])))
	if prob_a > prob_b: return [1,prob_a,prob_b]
	else: return [2,prob_a,prob_b]

def run(q):
	q = q.split()
	for i in range(len(q)):
		for sym in redundant: q[i] = q[i].replace(sym,'')
		q[i] = morph.parse(q[i])[0].normal_form
	return classify(q)

morph = pymorphy2.MorphAnalyzer()
inp_news = open("input_news", "r")
inp_science = open("input_science", "r")
news, sci = [], []
classes = [[],[]]

for line in inp_news: news.append(line)
for line in inp_science: sci.append(line)
pc1,pc2 = len(news)/(len(news) + len(sci)), len(sci)/(len(news) + len(sci)) 
redundant = [',', '–', '.', '…', '!', ':', '?', '»', '«', '(', ')']
for i in range(len(news)):
	for sym in redundant: news[i] = news[i].replace(sym,'')
	classes[0].append(news[i].split())
	for sym in redundant: sci[i] = sci[i].replace(sym,'')
	classes[1].append(sci[i].split())
classes[0] = norm(classes[0])
classes[1] = norm(classes[1])

que = ["Администрация социальной сети Facebook заблокировала страницу главы самопровозглашенной Донецкой народной республики (ДНР) Александра Захарченко.",
"В настоящий момент политика МВФ запрещает фонду предоставлять кредитование странам, которые имеют опыт просроченных платежей другим правительствам. А у Украины есть риск допустить дефолт по российским облигациям. В сентябре Киев запустил юридическую процедуру реструктуризации части своего госдолга.",
"Турция считает, что сделанное курдской партией 'Демократический союз' (PYD) заявление о создании автономии угрожает ее национальной безопасности. Анкара участвует в военной спецоперации в Сирии против боевиков 'Исламского государства'' (ИГ) в рамках коалиции под руководством США.",
"В Швеции неизвестные подожгли приют для беженцев-детей","В США в результате неожиданного возгорания двигателя пассажирского Boeing пострадал 21 человек. ЧП произошло вчера днем в международном аэропорту Форт-Лодердейла в штате Флорида. Первоначально власти сообщали о 15 раненых, однако, к несчастью, их оказалось больше.","Исследователям из Массачусетского технологического института удалось так модернизировать работу WiFi-передатчиков, что теперь с их помощью можно видеть сквозь стены.",
"Группа российских девушек, состоящая из восьми человек начала осуществлять земной полёт на Луну. Стоит сказать, что все участницы добровольно изъявили желание поучаствовать в наземном эксперименте получившем название «Луна-2015».",
"Сотрудники NASA сообщают, что космический зонд Cassini погрузится в струи фонтана, вырывающиеся со спутника Сатурна Энцелада. Такой маневр зонда будет сделан для исследования спутника на предмет возможности на нем жизни.",
"Ученые обнаружили эволюционные источники аллергии: биологи впервые систематически выявили сходство (на молекулярном уровне) между белками в пище и окружающей среде, вызывающими аллергические реакции, и белками многоклеточных паразитов.",
"Ученые Уральского федерального университета отправятся в Антарктиду для исследования метеоритов, сообщили Федеральному порталу 'Российское образование' в пресс-службе Министерства образования и науки РФ. Возглавит экспедицию профессор УрФУ Виктор Гроховский."]

answer = [1,1,1,1,1,2,2,2,2,2]
fp = 0
for i in range(len(que)):
	res = run(que[i])
	print(que[i])
	print(res)
	if res[0] == answer[i]:
		print(Fore.GREEN + "TRUE")
		fp += 1
	else: print (Fore.RED + "FALSE")
	print(Style.RESET_ALL)
	input()
print("Accuracy = " + str(fp/len(answer)))

