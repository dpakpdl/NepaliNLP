from sklearn.svm import SVC
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import sys
import os
import itertools
import pickle
import prepare_data


class SVM(object):
    def __init__(self, file, train_mode=True):
        self.clf = None
        self.file = file
        self.train_mode = train_mode
        self.training_data = prepare_data.train_set
        self.test_data = prepare_data.test_set
        self.training_sentences = prepare_data.train_sentences
        self.test_sentences = prepare_data.test_sentences
        self.sentences = prepare_data.sentences
        self.w2v_words = Word2Vec(self.sentences, size=10)
        self.tags = prepare_data.tags_list

    def __call__(self, *args, **kwargs):
        print(self.train_mode)
        if self.train_mode:
            print('inside')
            self.train()
        else:
            print('not')
            self.load()

    def pre_train(self):
        features_train = []
        labels = []
        for sent in tqdm(self.training_data):
            tagged_sent = [("", "<s>"), ("", "<s>")] + sent + [("", "</s>")]
            for j in range(2, len(tagged_sent) - 1):
                features_train.append(self.get_features(j, tagged_sent))
                labels.append(tagged_sent[j][1])
        return features_train, labels

    def train(self):
        data, labels = self.pre_train()
        self.clf = SVC(verbose=True, random_state=0, class_weight='balanced', probability=True)
        self.clf.fit(data, labels)
        print(self.clf)
        self.save()

    def save(self):
        with open(self.file, 'wb') as file:
            pickle.dump(self.clf, file)

    def load(self):
        with open(self.file, 'rb') as file:
            self.clf = pickle.load(file)
        print(self.clf)

    def predict(self):
        model_tagged_sentences = []
        for sent in tqdm(self.test_data):
            tagged_sent = [("", "<s>"), ("", "<s>")] + sent + [("", "</s>")]
            for j in range(0, len(sent)):
                features = np.array(self.get_features(j + 2, tagged_sent))
                tag = self.clf.predict(features.reshape(1, -1))
                tagged_sent[j + 2] = (sent[j][0], tag[0])
            model_tagged_sentences.append(tagged_sent[2:-1])
        return model_tagged_sentences

    def evaluate(self, model_tagged_sentences):
        corr_dict = defaultdict(int)
        wrong_dict = defaultdict(int)
        print(model_tagged_sentences[0])
        print(self.test_data[0])
        for i in range(len(self.test_data)):
            for j in range(len(self.test_data[i])):
                if self.test_data[i][j][1] == model_tagged_sentences[i][j][1]:
                    corr_dict[self.test_data[i][j][1]] += 1
                else:
                    wrong_dict[self.test_data[i][j][1]] += 1

        actual_labels = [tag for sentence in self.test_data for _, tag in sentence]
        predicted_labels = [tag for sentence in model_tagged_sentences for _, tag in sentence]
        print(confusion_matrix(actual_labels, predicted_labels))
        if self.train_mode:
            self.write_evaluation(actual_labels, predicted_labels)
        return actual_labels, predicted_labels

    def get_features(self, word_idx, sentence):
        features = []
        # words
        keys = self.w2v_words.wv.vocab.keys()
        n_prev_words = 1
        for i in reversed(range(0, n_prev_words + 1)):
            if sentence[word_idx - i][0] not in keys:
                features.append(np.zeros(self.w2v_words.vector_size))
            else:
                features.append(self.w2v_words.wv[sentence[word_idx - i][0]])
        n_next_words = 1
        for i in range(1, n_next_words + 1):
            if sentence[word_idx + i][0] not in keys:
                features.append(np.zeros(self.w2v_words.vector_size))
            else:
                features.append(self.w2v_words.wv[sentence[word_idx + i][0]])
        # previous_tags
        n_prev_tags = 2
        for i in reversed(range(1, n_prev_tags + 1)):
            tag = sentence[word_idx - i][1]
            features.append(self.one_hot_encoder(self.tags.index(tag), len(self.tags)))

        # flatten
        flat_list = [item for sublist in features for item in sublist]
        return flat_list

    @staticmethod
    def one_hot_encoder(number, length):
        if number > length:
            raise ValueError("Encoded number must be lower then one hot length")
        zero = np.zeros(length)
        zero[number] = 1
        return zero

    def write_evaluation(self, actual, predicted):
        out = open('data/SVM_metrics_{}_{}_train_sentences.txt'.format("NepaliPOS", len(self.training_sentences)), 'w')
        # Calculate metrics for each label, and find their average, weighted by support
        # (the number of true instances for each label).
        # This alters macro to account for label imbalance; it can result in an F-score
        # that is not between precision and recall.
        out.write(" Accuracy: {}\n".format(metrics.accuracy_score(actual, predicted)))

        # ability of the classifier not to label as positive a sample that is negative
        out.write(" Precision: {}\n".format(metrics.precision_score(actual, predicted, average='weighted')))

        # ability of the classifier to find all the positive samples.
        out.write(" Recall: {}\n".format(metrics.recall_score(actual, predicted, average='weighted')))

        out.write(" F1-Score: {}\n".format(metrics.f1_score(actual, predicted, average='weighted')))
        out.close()


if __name__ == "__main__":
    svm = SVM('data/svm_model.sav', train_mode=True)
    svm()
    print(svm.clf.coef_)
    model_tagged_sentences_ = svm.predict()
    # print(svm.test_data)
    # print(model_tagged_sentences_)
    svm.evaluate(model_tagged_sentences_)

