# -*- coding: utf-8 -*-

"""
@author: dpak
"""
from collections import Counter, defaultdict
import prepare_data
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
import re


class NepaliHMM(object):
    def __init__(self, training_data, test_data, vocabs):
        self.training_data = training_data
        self.test_data = test_data
        self.vocabs = vocabs
        self.training_sentences = []
        self.test_sentences = []
        self.tags = []
        self.words = []
        self.tag_unigrams = {}
        self.tag_bigrams = {}
        self.train_labels = []
        self.test_labels = []
        self.basic_model = None
        self.initialize_data()

    def initialize_data(self):
        self.tags = [tag for sentence in self.training_data for _, tag in sentence]
        self.words = [word for sentence in self.training_data for word, _ in sentence]
        for data in self.training_data:
            self.train_labels.append([tag for word, tag in data])
            self.training_sentences.append([word for word, tag in data])

        self.test_labels = []
        self.test_sentences = []
        for data in self.test_data:
            self.test_labels.append([tag for word, tag in data])
            self.test_sentences.append([word for word, tag in data])

        self.tag_unigrams = self.get_counts(self.tags)

        o = [(self.tags[i], self.tags[i + 1]) for i in range(0, len(self.tags) - 1)]

        self.tag_bigrams = self.get_counts(o)

    @staticmethod
    def get_counts(sequences):
        return Counter(sequences)

    @staticmethod
    def pair_counts(tags, words):
        d = defaultdict(lambda: defaultdict(int))
        for tag, word in zip(tags, words):
            d[tag][word] += 1
        return d

    def build(self):
        self.basic_model = HiddenMarkovModel(name="base-hmm-tagger")
        tags_count = self.get_counts(self.tags)
        tag_words_count = self.pair_counts(self.tags, self.words)

        starting_tag_list = [i[0] for i in self.train_labels]
        ending_tag_list = [i[-1] for i in self.train_labels]

        starting_tag_count = self.get_counts(starting_tag_list)  # the number of times a tag occured at the start
        ending_tag_count = self.get_counts(ending_tag_list)  # the number of times a tag occured at the end

        to_pass_states = []

        for tag, words_dict in tag_words_count.items():
            total = float(sum(words_dict.values()))
            distribution = {word: count / total for word, count in words_dict.items()}
            tag_emissions = DiscreteDistribution(distribution)
            tag_state = State(tag_emissions, name=tag)
            to_pass_states.append(tag_state)

        self.basic_model.add_states()

        start_prob = {}
        end_prob = {}
        for tag in self.tags:
            start_prob[tag] = starting_tag_count[tag] / tags_count[tag]
            end_prob[tag] = ending_tag_count[tag] / tags_count[tag]

        for tag_state in to_pass_states:
            self.basic_model.add_transition(self.basic_model.start, tag_state, start_prob[tag_state.name])

        for tag_state in to_pass_states:
            self.basic_model.add_transition(tag_state, self.basic_model.end, end_prob[tag_state.name])

        transition_prob_pair = {}

        for key in self.tag_bigrams.keys():
            transition_prob_pair[key] = self.tag_bigrams.get(key) / tags_count[key[0]]

        for tag_state in to_pass_states:
            for next_tag_state in to_pass_states:
                try:
                    self.basic_model.add_transition(tag_state, next_tag_state,
                                                    transition_prob_pair[(tag_state.name, next_tag_state.name)])
                except KeyError as err:
                    self.basic_model.add_transition(tag_state, next_tag_state, 0)

        self.basic_model.bake()

    def replace_unknown(self, sequence):
        return [w if w in self.vocabs else 'nan' for w in sequence]

    def simplify_decoding(self, sequence):
        _, state_path = self.basic_model.viterbi(self.replace_unknown(sequence))
        return [state[1].name for state in state_path[1:-1]]

    def accuracy(self, sequences, labels):
        correct = total_predictions = 0
        for observations, actual_tags in zip(sequences, labels):
            # The model.viterbi call in simplify_decoding will return None if the HMM
            # raises an error (for example, if a test sentence contains a word that
            # is out of vocabulary for the training set). Any exception counts the
            # full sentence as an error (which makes this a conservative estimate).
            try:
                most_likely_tags = self.simplify_decoding(observations)
                correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
            except Exception:
                pass
            total_predictions += len(observations)
        return correct / total_predictions

    def evaluate(self):
        hmm_training_acc = self.accuracy(self.training_sentences, self.train_labels)
        print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

        hmm_testing_acc = hmm.accuracy(self.test_sentences, self.test_labels)
        print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))
        for idx, key in enumerate(self.test_data[:2]):
            print("Sentence Key: {}\n".format(idx))
            print("Predicted labels:\n-----------------")
            words = [w for w, t in key]
            print(self.simplify_decoding(words))
            print("Actual labels:\n--------------")
            print([t for w, t in key])
            print("\n")


def get_user_input():
    return re.sub(r"([\.,?])", r" \1 ", input("\033[95mEnter your sentence ('exit' to exit): \033[0m"))


if __name__ == "__main__":
    # train_set, test_set = prepare_data.split(0.1)
    hmm = NepaliHMM(
        training_data=prepare_data.train_set,
        test_data=prepare_data.test_set,
        vocabs=prepare_data.words_list
    )
    hmm.build()
    hmm.evaluate()
    print('Initializing the tagger...')
    user_input = get_user_input()

    while user_input != 'exit':
        sentence_ = user_input.split(" ")
        try:
            tagged_output = hmm.simplify_decoding(sentence_)
            output = zip(sentence_, tagged_output)
            print("\nThe best tag sequence is:", dict(output))

        except Exception as ex:
            print(str(ex))

        user_input = get_user_input()

