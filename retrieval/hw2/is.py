import pymorphy2
from operator import itemgetter
morph = pymorphy2.MorphAnalyzer()

inp = open('input.txt', 'r')
out = open('output.txt', 'w')
text = list(inp)
text_s = []
redundant = [',', '–', '.', '…', '!', ':', '?', '»', '«', '(', ')']
for line in text:
	for sym in redundant: line = line.replace(sym,'')
	text_s.append(line.split())

text_s = sum(text_s, [])

dict = {}

for w in text_s:
	p = morph.parse(w)[0].normal_form
	if p in dict:
		dict[p] += 1
	else: 
		dict[p] = 1

for k, v in sorted(dict.items(), key=itemgetter(1), reverse = True):
    out.write(''.join((k, ' ', str(v), '\n'))) 