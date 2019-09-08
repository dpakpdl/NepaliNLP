import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(THIS_DIR))


def get_data_path():
    return os.path.join(os.path.dirname(THIS_DIR), 'data')


def get_word_vocab_path():
    data_path = get_data_path()
    return os.path.join(data_path, "vocabs", "words.txt")


def get_char_vocab_path():
    data_path = get_data_path()
    return os.path.join(data_path, "vocabs", "chars.txt")


def get_tag_vocab_path():
    data_path = get_data_path()
    return os.path.join(data_path, "vocabs", "tags.txt")


def get_training_data_path():
    data_path = get_data_path()
    return os.path.join(data_path, "UrduNepaliEnglish", "pos1.csv")


def get_testing_data_path():
    data_path = get_data_path()
    return os.path.join(data_path, "UrduNepaliEnglish", "pos3.csv")


def get_results_path():
    return os.path.join(os.path.dirname(THIS_DIR), 'results')


def get_saved_model_path():
    return os.path.join(os.path.dirname(THIS_DIR), 'model', 'saves')


def get_merged_data_path():
    data_path = get_data_path()
    return os.path.join(data_path, "UrduNepaliEnglish", "merged.csv")
