class CharacterGetter(object):
    def __init__(self, words):
        self.words = words
        self.characters = set([w_i for w in self.words for w_i in w])
        self.number_of_characters = len(self.characters)
        self.char2idx = self.set_char2idx()

    def set_char2idx(self):
        char2idx = {c: i + 2 for i, c in enumerate(self.characters)}
        char2idx["UNK"] = 1
        char2idx["PAD"] = 0
        return char2idx
