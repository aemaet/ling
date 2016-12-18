exp = '"Русгидро" "Росгеология" "Транснефть" получат предложения перенести свои главные офисы на Дальний Восток.'
s1 = 'Компаниям "Русгидро" "Транснефть" и "Росгеология" предложили подумать о переезде на Дальний Восток.'
s2 = 'По словам вице-премьера по Дальнему Востоку переезд может как-то затронуть "РусГидро" "Транснефть" и "Росгеологию"'

def rouge(x,y):
	a = len(x.intersection(y))
	#print(x.intersection(y),x,y)
	b = len(y)
	return a/b

setE = set(exp.split())
set1 = set(s1.split())
set2 = set(s2.split())
res = [rouge(set1,setE),rouge(set2,setE)]
print(res)

#ручной счет - 6/12, 5/12 