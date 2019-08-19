import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)
sys.path.append(os.path.dirname(THIS_DIR))

from data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word
from general_utils import get_logger
from helper import utility


class Config(object):
    def __init__(self, load=True):
        """Initialize hyper parameters and load vocabs
        Args:
            load: (bool) if True, load embeddings into
                np array, else None
        """
        self.vocab_words = None
        self.vocab_chars = None
        self.vocab_tags = None
        self.number_of_chars = None
        self.number_of_words = None
        self.number_of_tags = None
        self.processing_tag = None
        self.processing_word = None
        self.embeddings = None

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings
        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)
        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        #
        self.number_of_words = len(self.vocab_words)
        self.number_of_chars = len(self.vocab_chars)
        self.number_of_tags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed) if self.use_pre_trained else None)

    # general config
    dir_output = utility.get_results_path() + "/" + "test"
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "/log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = utility.get_data_path() + "/word2vec/nepali_embeddings_word2vec.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = utility.get_data_path() + "/word2vec/nepali_embeddings_word2vec.{}d.trimmed.npz".format(dim_word)
    use_pre_trained = False

    # dataset

    filename_test = utility.get_testing_data_path()
    filename_train = utility.get_training_data_path()

    max_iter = 50  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = utility.get_word_vocab_path()
    filename_tags = utility.get_tag_vocab_path()
    filename_chars = utility.get_char_vocab_path()

    # training
    train_embeddings = False
    nepochs = 1
    dropout = 0.5
    batch_size = 20
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.90
    epoch_drop = 2  # Step Decay: per # epochs to apply lr_decay
    clip = 5  # if negative, no clipping
    nepoch_no_imprv = 5  # number of epoch with no improvement, used for EarlyStopping

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU

    train_validate_split = 0.1
