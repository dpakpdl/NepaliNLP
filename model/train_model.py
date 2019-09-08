#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import warnings

from keras.utils import plot_model

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)

from data_utils import TrainDevData
from bilstm_crf import BLSTMCRF
from config import Config

warnings.filterwarnings('ignore')


def main():
    # create instance of config
    config = Config()

    # build model
    model = BLSTMCRF(config)
    model.build()
    model.compile(optimizer=model.get_optimizer(), loss=model.get_loss(), metrics=model.get_metrics())
    print("Number of Tags: {0}".format(config.number_of_tags))
    print("Number of Characters: {0}".format(config.number_of_chars))
    print("Number of Words: {0}".format(config.number_of_words))

    # Loading weights
    # model.load_weights("./saves/model12.h5", by_name=True)

    # create datasets
    data = TrainDevData(
        config.filename_train,
        config.processing_word,
        config.processing_tag,
        config.max_iter,
        config.train_validate_split
    )

    print('train={}'.format(sum(1 for x in data.train)))
    print('validate={}'.format(sum(1 for x in data.validate)))
    model.summary()
    # train model
    start_time = time.time()
    history = model.train(data.train, data.validate)
    end_time = time.time()
    print("time taken to train: {0}".format(end_time - start_time))
    print(plot_model(model, to_file='./saves/model.png'))

    # Save model
    # model.model.save_weights('./saves/model_no_crf_20e_lr_0.001_new_presentation_day.h5')
    model.plot_history(history)
    # test predictions

    sentences = "पेस्की रकम लिई ठेक्का अलपत्र पार्ने क्रम बढे पछि नियन्त्रण गर्न सरकार ले कडाइ गरेको छ ।"

    words = sentences.split(' ')
    pred = model.predict_words(words)
    print(list(zip(words, pred)))


if __name__ == "__main__":
    main()
