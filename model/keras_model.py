import os
import sys
import warnings

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)

from callbacks import F1score
from data_utils import minibatches, pad_sequences

warnings.filterwarnings('ignore')


class BaseKerasModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger
        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings
        """
        self.config = config
        self.logger = config.logger
        self.model = None
        self.sess = None
        self.saver = None

    def batch_iter(self, train, batch_size, return_lengths=False):
        """
        Creates a batch generator for the dataset
        :param train: Dataset
        :param batch_size: Batch Size
        :param return_lengths: If True, generator returns sequence lengths. Used masking data during the evaluation step
        :return: (number of batches in dataset, data generator)
        """
        # number_of_batches = (len(train) + batch_size - 1) // batch_size
        number_of_batches = (sum(1 for x in train) + batch_size - 1) // batch_size

        def _get_data(words, labels):
            # perform padding of the given data
            char_ids = list()
            if self.config.use_chars:
                char_ids, word_ids = zip(*words)
                word_ids, sequence_lengths = pad_sequences(word_ids, 0)
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            else:
                word_ids, sequence_lengths = pad_sequences(words, 0)

            if labels:
                labels, _ = pad_sequences(labels, 0)
                # Change labels to one-hot
                labels = [to_categorical(label, num_classes=self.config.number_of_tags + 1) for label in labels]

            # build dictionary
            inputs = {
                "word_ids": np.asarray(word_ids),
            }
            if self.config.use_chars:
                inputs["char_ids"] = np.asarray(char_ids)

            if return_lengths:
                return inputs, np.asarray(labels), sequence_lengths
            else:
                return inputs, np.asarray(labels)

        def data_generator():
            while True:
                for i, (words, labels) in enumerate(minibatches(train, batch_size)):
                    results = _get_data(words, labels)
                    yield results

        return number_of_batches, data_generator()

    def train(self, train, dev, show_history=False):
        batch_size = self.config.batch_size

        number_of_training_batches, train_generator = self.batch_iter(train, batch_size)
        number_of_validation_batches, dev_generator = self.batch_iter(dev, batch_size)

        _, f1_generator = self.batch_iter(dev, batch_size, return_lengths=True)
        f1 = F1score(f1_generator, number_of_validation_batches, self.run_evaluate)
        callbacks = self.gen_callbacks([f1])

        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=number_of_training_batches,
                                           validation_data=dev_generator,
                                           validation_steps=number_of_validation_batches,
                                           epochs=self.config.nepochs,
                                           callbacks=callbacks,
                                           use_multiprocessing=False)  # , nbatches_train

        if show_history:
            print(history.history['f1'])
        return history

    def predict_words(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        char_ids, word_ids = words
        word_ids = np.asarray(word_ids)
        s = word_ids.shape
        word_ids = word_ids.reshape(-1, s[0])
        inputs = [word_ids]

        if self.config.use_chars:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0)
            char_ids = np.asarray(char_ids)
            s = char_ids.shape
            char_ids = char_ids.reshape(-1, s[0], s[1])
            inputs.append(char_ids)

        one_hot_preds = self.model.predict_on_batch(inputs)
        # print("One hot preds: ", one_hot_preds)
        one_hot_preds = [a.flatten() for a in one_hot_preds.squeeze()]
        # Squeeze to remove unnecessary 1st dimension for batch size

        # print("One hot preds: ", one_hot_preds)
        pred_ids = np.argmax(one_hot_preds, axis=1)
        # print("Pred ids: ", pred_ids)

        predictions = [self.idx_to_tag[idx] for idx in pred_ids]
        return predictions

    def run_evaluate(self, data_generator, steps_per_epoch):
        label_true = []
        label_pred = []
        for i in range(steps_per_epoch):
            x_true, y_true, sequence_lengths = next(data_generator)
            y_pred = self.model.predict_on_batch(x_true)
            for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                lab = np.argmax(lab, axis=1)
                lab_pred = np.argmax(lab_pred, axis=1)

                label_true.extend(lab)
                label_pred.extend(lab_pred)
        label_true = np.asarray(label_true)
        # print('label==={0}'.format(label_true))
        label_pred = np.asarray(label_pred)
        # print('pred==={0}'.format(label_pred))

        micro_score = f1_score(label_true, label_pred, average='micro', labels=np.unique(label_pred))
        print(' - micro f1: {:04.2f}'.format(micro_score * 100))

        macro_score = f1_score(label_true, label_pred, average='macro', labels=np.unique(label_pred))
        print(' - macro f1: {:04.2f}'.format(macro_score * 100))

        weighted_score = f1_score(label_true, label_pred, average='weighted', labels=np.unique(label_pred))
        print(' - weighted f1: {:04.2f}'.format(weighted_score * 100))

        print('report: {0}'.format(classification_report(label_true, label_pred)))

        self._evaluate(label_true, label_pred)

        return micro_score, macro_score, weighted_score

    @staticmethod
    def _evaluate(label_true, label_predicted):
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(label_true, label_predicted)
        print('Accuracy: %f' % accuracy)

        # precision tp / (tp + fp)
        precision = precision_score(label_true, label_predicted, average='micro')
        # average = None will return the precision scores for each class
        # average='micro' will return the total ratio of tp/(tp + fp)
        print('Precision: %f' % precision)

        # recall: tp / (tp + fn)
        recall = recall_score(label_true, label_predicted, average="micro")
        print('Recall: %f' % recall)

        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(label_true, label_predicted, average="micro")
        print('F1 score: %f' % f1)

        # kappa
        # kappa = cohen_kappa_score(label_true, label_predicted)
        # print('Cohens kappa: %f' % kappa)

        # confusion matrix
        matrix = confusion_matrix(label_true, label_predicted)
        print(matrix)

    def get_loss(self):
        return self._loss

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_optimizer(self):
        return self._optimizer
