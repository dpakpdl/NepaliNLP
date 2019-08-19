#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)

from data_utils import TrainDevData
from bilstm_crf import BLSTMCRF
from config import Config


def main():
    # create instance of config
    config = Config()

    model = BLSTMCRF(config)

    model.build()
    model.compile(optimizer=model.get_optimizer(), loss=model.get_loss())

    # model.load_weights('./saves/model_no_crf_20e_lr_0.001_new.h5')

    test = TrainDevData(config.filename_train, config.processing_word,
                        config.processing_tag, config.max_iter, config.train_validate_split)
    total = 0
    for a, w in test.validate:
        total += len(a)

    print(total)

    batch_size = config.batch_size
    nbatches_test, test_generator = model.batch_iter(test, batch_size, return_lengths=True)

    model.run_evaluate(test_generator, nbatches_test)

    # test predictions
    # words = "मेरो देश नेपाल निकै प्यारो लाग्नु को कारण हरु धेरै भएको बिदेशी हरु ले बताए ?"
    words = "ठेकेदार ले सम्बद्ध क्षेत्र मा पेस्की रकम खर्च नगर्ने देखिएकाले " \
            "त्यस लाई निरुत्साहित गर्नु पर्ने सिटौला ले बताए ।"
    words = words.split(" ")
    pred = model.predict_words(words)
    print(list(zip(words, pred)))


if __name__ == "__main__":
    main()
