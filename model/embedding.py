# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(THIS_DIR))

from pre_processing.get_tagset import read
from pre_processing.get_characters import CharacterGetter
from pre_processing.get_sentences import SentenceGetter
from pre_processing.get_words import WordTagGetter
from pre_processing.pre_process import get_char2idx_sentences

data = read()
sentence_getter = SentenceGetter(data)
sentences = sentence_getter.sentences

word_tag_getter = WordTagGetter(data)
words = word_tag_getter.words
word2idx = word_tag_getter.word2idx
n_words = word_tag_getter.number_of_words
n_tags = word_tag_getter.number_of_tags
idx2word = word_tag_getter.idx2word
tag2idx = word_tag_getter.tag2idx
idx2tag = word_tag_getter.idx2tag

max_len = 200
max_len_char = 50

# sentences = sentences[:100]
print(sentences)
X_word = [[word2idx[w[0]] for w in s] for s in sentences]

X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

character_getter = CharacterGetter(words)
characters = character_getter.characters
char2idx = character_getter.char2idx
n_chars = character_getter.number_of_characters

X_char = get_char2idx_sentences(sentences, max_len_char, max_len, char2idx)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
# print(X_word)
# print(y)
X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)

# print(X_word_tr[0])
# print(X_word_te[0])
# print(y_tr[0])
# print(y_te[0])

# input and embedding for words
word_in = Input(shape=(max_len,))
print(word_in)
emb_word = Embedding(input_dim=n_words + 2, output_dim=20, input_length=max_len, mask_zero=True)(word_in)

print(emb_word)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10, input_length=max_len_char, mask_zero=True))(
    char_in)

# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
print(x)
x = SpatialDropout1D(0.3)(x)
print(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.5))(x)

out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

model = Model([word_in, char_in], out)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

model.summary()

history = model.fit([X_word_tr, np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
                    batch_size=32, epochs=10, validation_split=0.1, verbose=1)
print(history)
y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])

i = 1925
print(len(y_pred))
print(y_pred[0])
p = np.argmax(y_pred[0], axis=-1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_word_te[0], y_te[0], p):
    if w != 0:
        print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))
