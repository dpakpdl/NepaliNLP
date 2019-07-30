# -*- coding: utf-8 -*-
import numpy as np
import logging as log


def get_char2idx_sentences(sentences: list, max_len_char: int, max_len: int, char2idx: dict) -> list:
    char2idx_from_sentences = []
    for sentence in sentences:
        sent_seq = _get_char2idx_sentence(sentence, max_len_char, max_len, char2idx)
        char2idx_from_sentences.append(np.array(sent_seq))
    return char2idx_from_sentences


def _get_char2idx_sentence(sentence, max_len_char, max_len, char2idx):
    sent_seq = list()
    for i in range(max_len):
        word_seq = list()
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except Exception as ex:
                log.info(ex)
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    return sent_seq
