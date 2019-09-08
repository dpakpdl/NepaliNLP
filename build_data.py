from model.config import Config
from model.data_utils import TrainDevData, get_vocabs, UNK, get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe/FastText vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.

    config: (instance of Config) has attributes like hyper-params...

    """
    # 1. get config and processing of words
    config = Config(load=False)

    # 2. Get processing word generator
    processing_word = get_processing_word()

    # 3. Generators
    test = TrainDevData(config.filename_test, processing_word)
    train = TrainDevData(config.filename_train, processing_word)

    # 4. Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    # 5. Get a vocab set for words in both vocab_words and vocab_glove
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    # vocab.add(NUM)

    # 6. Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # 7. Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                 config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = TrainDevData(config.filename_train)
    test = TrainDevData(config.filename_test)
    vocab_chars_train = get_char_vocab(train)
    vocab_chars_test = get_char_vocab(test)
    vocab_chars = vocab_chars_test.union(vocab_chars_train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
