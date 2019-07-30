# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
# Load vectors
model = KeyedVectors.load_word2vec_format('nepali_embeddings_word2vec.txt', binary=False)
# find similarity between words
model.similarity('फेसबूक', 'इन्स्टाग्राम')

# most similar words
model.most_similar('ठमेल')

# try some linear algebra maths with Nepali words
model.most_similar(positive=['', ''], negative=[''], topn=1)
