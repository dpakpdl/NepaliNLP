#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import gensim
# model = gensim.models.Word2Vec.load('word2vec/cc.ne.300.vec')
# words = model.most_similar(positive=[], negative=[], topn=10)
from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('word2vec/cc.ne.300.bin')

print(model.most_similar(positive=['थोपा'], negative=[], topn=10))

# print(model.similarity('teacher', 'teaches'))

# import io


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        # data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = tokens[1:]
        count +=1
        if count >=200:
            break
    return data



# print(load_vectors('word2vec/cc.ne.300.vec'))

print (map(float, [1.0, 2.1, -0.3]))