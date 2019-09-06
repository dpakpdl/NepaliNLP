import sys
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(THIS_DIR)))

from NeuralNetwork import get_tagset
from NeuralNetwork.get_sentences import SentenceGetter
from NeuralNetwork.get_words import WordTagGetter

training_data = get_tagset.read("data/pos1.csv")
sentence_getter = SentenceGetter(training_data)
train_set = sentence_getter.tagged_sentences
train_sentences = sentence_getter.sentences
word_tag_getter = WordTagGetter(training_data)
train_tags_list = word_tag_getter.tags
n_train_sents = len(train_sentences)


test_data = get_tagset.read("data/pos3.csv")
sentence_getter = SentenceGetter(test_data)
test_set = sentence_getter.tagged_sentences
test_sentences = sentence_getter.sentences
word_tag_getter = WordTagGetter(training_data)
test_tags_list = word_tag_getter.tags
n_test_sents = len(test_sentences)

tags_list = train_tags_list + test_tags_list
tags_list.append("<s>")
tags_list.append("</s>")

tagged_sentences = train_set + test_set
sentences = train_sentences + test_sentences
print('Number of sentences: {}'.format(n_test_sents + n_train_sents))
print('Train set Length: {}'.format(n_train_sents))
print('Test set Length: {}'.format(n_test_sents))
