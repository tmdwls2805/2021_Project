from mecab import Mecab
import pandas as pd

st = []
me = Mecab()

f = open('./data/myd/train/seq.in', 'r', encoding='utf-8')
line = f.readlines()
for i in line:
    st.append(i)

fword = []
fpos = []
word_pos = []
for i in range(1):
    for word in st:
        word_set = me.pos(word)
        word_list = []
        pos_list = []
        for w in word_set:
            word_list.append(w[0])
            pos_list.append(w[1])
        fword.append(word_list)
        fpos.append(pos_list)
b = []
for i in fword:
    a = ' '.join(i)
    b.append(a)
print(b)
ffword = open('seq.in', mode='wt', encoding='utf-8')

for i in b:
    ffword.write(i+'\n')
ffword.close()
