import os
import sys
import warnings
from functools import partial

from bayes_opt import BayesianOptimization
from keras import optimizers
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

# from keras import losses

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(THIS_DIR))
from model.bilstm_crf import BLSTMCRF
from model.config import Config
from model.data_utils import TrainDevData

verbose = 1

warnings.filterwarnings('ignore')


def get_model():
    model = BLSTMCRF(Config())
    model.build()
    return model


def fit_with(verbose, dropout2_rate, lr, epoch_drop, lr_decay):
    # Create the model using a specified hyper parameters.
    print("==================================================>")
    print(dropout2_rate, lr, epoch_drop, lr_decay)
    print("<==================================================")
    model = get_model()
    model.config.dropout = dropout2_rate
    model.config.lr = lr
    model.config.epoch_drop = epoch_drop
    model.config.lr_decay = lr_decay

    # Train the model for a specified number of epochs.
    optimizer = optimizers.Adam(lr=lr)

    model.compile(loss=crf_loss,
                  optimizer=optimizer,
                  metrics=[crf_accuracy])

    # create data-sets
    data = TrainDevData(
        model.config.filename_train,
        model.config.processing_word,
        model.config.processing_tag,
        model.config.max_iter,
        model.config.train_validate_split
    )
    # train model
    history = model.train(data.train, data.validate)
    accuracy = max(history.history['val_crf_accuracy'])
    return accuracy


fit_with_partial = partial(fit_with, verbose)

# Bounded region of parameter space
pbounds = {'dropout2_rate': (0.4, 0.6), 'lr': (1e-3, 1e-2), "epoch_drop": (1, 4), "lr_decay": (0.8, 0.9)}

b_optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

b_optimizer.maximize(init_points=10, n_iter=10)

for i, res in enumerate(b_optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(b_optimizer.max)
