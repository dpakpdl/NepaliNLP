import pandas as pd


class WordTagGetter(object):
    def __init__(self, tag_set):
        self.data = tag_set
        self.words = list(set(self.data["word"].values))
        self.number_of_words = len(self.words)

        self.tags = list(set(self.data["pos"].values))
        self.number_of_tags = len(self.tags)
        self.word2idx = self.set_word2idx()
        self.idx2word = self.set_idx2word()
        self.tag2idx = self.set_tag2idx()
        self.idx2tag = self.set_idx2tag()

    def set_word2idx(self):
        word2idx = {w: i + 2 for i, w in enumerate(self.words)}
        word2idx["UNK"] = 1
        word2idx["PAD"] = 0
        return word2idx

    def set_idx2word(self):
        return {i: w for w, i in self.word2idx.items()}

    def set_tag2idx(self):
        tag2idx = {t: i + 1 for i, t in enumerate(self.tags)}
        tag2idx["PAD"] = 0
        return tag2idx

    def set_idx2tag(self):
        return {i: w for w, i in self.tag2idx.items()}


def main():
    data = pd.read_csv("sentences.csv", encoding="utf-8")
    data = data.fillna(method="ffill")
    print(data.tail(10))
    getter = WordTagGetter(data)
    print(getter.idx2tag)
    # max_len = 75
    # max_len_char = 10


if __name__ == "__main__":
    main()
