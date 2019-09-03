import pandas as pd


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.grouped = self.data.groupby("sentence").apply(self.aggregate)
        self.grouped_words = self.data.groupby("sentence").apply(self.aggregate_words)
        self.tagged_sentences = [s for s in self.grouped]
        self.sentences = [s for s in self.grouped_words]

    @staticmethod
    def aggregate(s):
        return [(word, pos) for word, pos in zip(s["word"].values.tolist(), s["pos"].values.tolist())]

    @staticmethod
    def aggregate_words(s):
        return [word for word in s["word"].values.tolist()]

    def get_next(self):
        try:
            s = self.grouped["sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except Exception as ex:
            print (ex)
            return None


def main():
    data = pd.read_csv("sentences.csv", encoding="utf-8")
    data = data.fillna(method="ffill")
    print(data.tail(10))
    getter = SentenceGetter(data)
    print(getter.tagged_sentences)


if __name__ == "__main__":
    main()
