#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

filename = "/Users/deepakpaudel/mycodes/NepaliNLP/Nepali-Tagged-Corpus/01ne_pos.txt"
output_file = "./pos2.csv"


def parse_sentence(sentence, index):
    words = sentence.strip().split('>')

    words = list(filter(None, words))
    tagged_words = list()

    for word in words:
        try:
            word_, tag = word.split('<')
            tagged_words.append({"sentence": index, "word": word_, "pos": tag})
        except ValueError as er:
            print(word)
            if "\"" in word:
                print('here')
                pass
            else:
                pass
    return tagged_words


def parse_corpus(total):
    parsed_sentences = list()
    for index, sentence in enumerate(total):
        parsed_sentences = parsed_sentences + parse_sentence(sentence, index+1917)
    return parsed_sentences


def write_to_csv(sentences):
    keys = sentences[0].keys()
    print(list(keys))
    with open(output_file, 'w') as csvfile:
        dict_writer = csv.DictWriter(csvfile, keys)
        dict_writer.writeheader()
        dict_writer.writerows(sentences)


with open(filename, "r") as in_file:
    total = in_file.readlines()
content = [x.strip() for x in total]
total = parse_corpus(content)
write_to_csv(total)

# print(parse_sentence(content[0]))
print(len(content))



