import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# shared global variables to be imported from model also
UNK = "UNK"
NONE = "O"


class MyIOError(Exception):
    def __init__(self, filename):
        message = """ERROR: Unable to locate file {}.FIX: Have you tried running python build_data.py first? 
        This will build vocab file from your train, test and dev sets and trimm your word vectors.""".format(filename)
        super(MyIOError, self).__init__(message)


class TrainDevData(object):
    """Class that iterates over dataset. __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None, optional pre-processing is applied
    Example:
        ```python
        data = Dataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None, split=None):
        """
        :param filename: path to the file
        :param processing_word: function that takes a word as input
        :param processing_tag: function that takes a tag as input
        :param max_iter: max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.grouped = None
        self.train = None
        self.validate = None
        self.train_validate_split = split
        self._call()

    def _call(self):
        data = pd.read_csv(self.filename, encoding="utf-8")
        self.grouped = data.groupby("sentence").apply(self.aggregate)
        if self.max_iter:
            self.grouped = self.grouped[:self.max_iter]
        if self.train_validate_split:
            self.train, self.validate = train_test_split(self.grouped, test_size=self.train_validate_split)

    def __iter__(self):
        for s in self.grouped:
            yield s

    def aggregate(self, s):
        words = list()
        tags = list()
        for word, tag in zip(s["word"].values.tolist(), s["pos"].values.tolist()):
            if self.processing_word is not None:
                word = self.processing_word(word)
            if self.processing_tag is not None:
                tag = self.processing_tag(tag)
            words += [word]
            tags += [tag]
        return words, tags

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects
    Args:
        datasets: a list of dataset objects
    Returns:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            words = [word.strip() for word in words if type(word) != float]
            vocab_words.update(words)
            tags = [tag.strip().upper() for tag in tags]
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    print("Building char vocab...")
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            word = word.strip() if type(word) != float else word
            vocab_char.update(str(word))
    print("- done. {} tokens".format(len(vocab_char)))
    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file
    Args:
        filename: path to the glove vectors
    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding="utf8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file
    Writes one word per line.
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            word = word.strip()
            if len(word) == 0:
                continue
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file
    Args:
        filename: (string) the format of the file must be one word per line.
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding="utf8") as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    print(embeddings)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None, chars=False, allow_unk=True):
    """
    Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.
    :param vocab_words: dict[word] = idx
    :param vocab_chars:
    :param chars:
    :param allow_unk:
    :return:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """

    def my_lambda(word):
        word = str(word)
        char_ids = []
        # 0. get chars of words
        if vocab_chars is not None and chars is True:
            char_ids = get_characters_from_words(word, vocab_chars)

        # 2. get id of word
        if vocab_words is not None:
            word = get_word_ids_from_word_vocab(word, vocab_words, allow_unk)

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars is True:
            return char_ids, word
        else:
            return word

    return my_lambda


def get_word_ids_from_word_vocab(word, vocab_words, allow_unk):
    word = word.strip()
    if word in vocab_words:
        return vocab_words[word]
    else:
        if allow_unk:
            return vocab_words[UNK]
        else:
            print(word)
            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")


def get_characters_from_words(word, vocab_chars):
    char_ids = []
    for char in str(word):
        # ignore chars out of vocabulary
        if char in vocab_chars:
            char_ids += [vocab_chars[char]]
    return char_ids


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)
        return sequence_padded, sequence_length
    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

        return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Yields:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch
