import math
import os
import sys

import keras.backend as K
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Dropout, Lambda, Concatenate
from keras.models import Model
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)

from callbacks import LossHistory
from base_model import BaseModel


class BLSTMCRF(BaseModel):

    def __init__(self, config):
        super(BLSTMCRF, self).__init__(config)
        self._metrics = ["acc"]
        self._loss = None  # losses.sparse_categorical_crossentropy
        self._optimizer = optimizers.Adam(lr=self.config.lr)
        # self._optimizer = optimizers.SGD(lr=self.config.lr, momentum=0.9, decay=0.05, clipvalue=self.config.clip)
        self.idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}
        self.model = None

    def build(self):
        inputs = []  # Create input for Model

        # build word embeddings
        input_words = Input(shape=(None,), dtype='int32', name='word_ids')
        inputs.append(input_words)
        if self.config.embeddings is None:
            word_embeddings = Embedding(input_dim=self.config.number_of_words,
                                        output_dim=self.config.dim_word,
                                        mask_zero=True,
                                        name="word_embeddings")(input_words)
        else:
            word_embeddings = Embedding(input_dim=self.config.number_of_words,
                                        output_dim=self.config.dim_word,
                                        mask_zero=True,
                                        weights=[self.config.embeddings],
                                        trainable=self.config.train_embeddings,
                                        name="word_embeddings")(input_words)

        # build character based word embedding
        if self.config.use_chars:
            input_chars = Input(batch_shape=(None, None, None), dtype='int32', name='char_ids')
            inputs.append(input_chars)
            char_embeddings = Embedding(input_dim=self.config.number_of_chars,
                                        output_dim=self.config.dim_char,
                                        mask_zero=True,
                                        name='char_embeddings')(input_chars)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config.dim_char)))(char_embeddings)

            # BiLSTM for char_embeddings
            fwd_state = LSTM(self.config.hidden_size_char, return_state=True,
                             name='forward_character_lstm')(char_embeddings)[-2]

            bwd_state = LSTM(self.config.hidden_size_char, return_state=True, go_backwards=True,
                             name='backward_character_lstm')(char_embeddings)[-2]

            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])

            # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.config.hidden_size_char]))(
                char_embeddings)

            # combine characters and word
            word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self.config.dropout)(word_embeddings)
        encoded_text = Bidirectional(LSTM(units=self.config.hidden_size_lstm, return_sequences=True),
                                     name="bidirectional")(word_embeddings)
        encoded_text = Dropout(self.config.dropout)(encoded_text)

        if self.config.use_crf:
            # spare_target = False uses categorical_crossentropy
            # this loss function is used for one hot encoded targets; see to_categorical
            # otherwise sparse_categorical_crossentropy is used as loss function
            crf = CRF(self.config.number_of_tags, sparse_target=False)
            self._loss = crf.loss_function
            self._metrics = [crf_accuracy]
            pred = crf(encoded_text)

        else:
            self._loss = 'categorical_crossentropy'
            pred = Dense(self.config.number_of_tags, activation='softmax')(encoded_text)

        self.model = Model(inputs, pred)

    def gen_callbacks(self, callbacks_list):
        lrate = LearningRateScheduler(self.step_decay)
        callbacks_list.append(lrate)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.nepoch_no_imprv
        )
        callbacks_list.append(early_stop)
        loss_history = LossHistory(self.step_decay)
        callbacks_list.append(loss_history)
        return callbacks_list

    def step_decay(self, epoch):
        initial_lr = self.config.lr
        decay = self.config.lr_decay
        epochs_drop = self.config.epoch_drop
        # lrate = initial_lrate * math.pow(decay, math.floor((1 + epoch) / epochs_drop))
        return initial_lr * math.pow(decay, math.floor(epoch / epochs_drop))

    @staticmethod
    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # summarize history for loss
        ax1.plot(history.history['loss'], label="train")
        ax1.plot(history.history['val_loss'], label="validate")
        ax1.set_title("model loss")
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(loc='upper left')

        # summarize history for accuracy
        ax2.plot(history.history['crf_accuracy'], label="train")
        ax2.plot(history.history['val_crf_accuracy'], label="validate")
        ax2.set_title('model accuracy')
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')
        ax2.legend(loc='upper left')
        plt.show()

    def get_metrics(self):
        return self._metrics
