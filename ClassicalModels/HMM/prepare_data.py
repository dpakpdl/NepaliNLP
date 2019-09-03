import sys
import os
from sklearn.model_selection import train_test_split
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(THIS_DIR)))

from pre_processing import get_tagset
from pre_processing.get_sentences import SentenceGetter
from pre_processing.get_words import WordTagGetter
from helper.utility import get_training_data_path, get_testing_data_path

training_data = get_tagset.read(get_training_data_path())
sentence_getter = SentenceGetter(training_data)
train_set = sentence_getter.tagged_sentences
word_tag_getter = WordTagGetter(training_data)
train_words_list = word_tag_getter.words


test_data = get_tagset.read(get_testing_data_path())
sentence_getter = SentenceGetter(test_data)
test_set = sentence_getter.tagged_sentences

word_tag_getter = WordTagGetter(training_data)
test_words_list = word_tag_getter.words

words_list = train_words_list + test_words_list

tagged_sentences = train_set + test_set


def split(percentage=0.1):
    labels = []
    t_sentences = []
    for sentence in tagged_sentences:
        labels.append([tag for _, tag in sentence])
        t_sentences.append([word for word, _ in sentence])

    train, test, label_train, label_test = train_test_split(t_sentences, labels, test_size=percentage)
    print('Number of sentences: {}'.format(len(tagged_sentences)))
    print('Train set Length: {}'.format(len(train)))
    print('Test set Length: {}'.format(len(test)))

    final_train = list()
    for x, y in zip(train, label_train):
        final_train.append([(w, t) for w, t in zip(x, y)])

    final_test = list()
    for x, y in zip(test, label_test):
        final_test.append([(w, t) for w, t in zip(x, y)])
    return final_train, final_test
