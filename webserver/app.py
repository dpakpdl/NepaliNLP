import json
import logging as log
import os
import sys
import warnings

from flask import Flask, render_template, request, jsonify, current_app
from rq import Queue
from rq.job import Job
from worker import conn

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(THIS_DIR))

from model.config import Config
from model.bilstm_crf import BLSTMCRF
from helper import utility

warnings.filterwarnings('ignore')

app = Flask(__name__)

q = Queue(connection=conn)


def _configure_logging():
    level = 2
    log.root.setLevel(level)
    log.disable(level - 1)
    log.root.addHandler(_create_console_logger())


def _create_console_logger():
    stderr_format = "%(levelname)s: %(message)s"
    stderr_handler = log.StreamHandler()
    stderr_handler.setFormatter(log.Formatter(stderr_format))
    return stderr_handler


def predict(sentence,):
    words = sentence.split(" ")
    pred = load_module().predict_words(words)
    return dict(zip(words, pred))


@app.before_first_request
def setup_logger():
    _configure_logging()


def load_module():
    config = Config()
    my_model = BLSTMCRF(config)
    my_model.build()
    my_model.compile(optimizer=my_model.get_optimizer(), loss=my_model.get_loss())

    saved_model_path = utility.get_saved_model_path() + '/model_no_crf_20e_lr_0.001_new.h5'

    my_model.load_weights(saved_model_path)

    return my_model


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def get_counts():
    # get url
    data = json.loads(request.data.decode())
    sentence = data["sentence"]

    # start job
    job = q.enqueue_call(
        func=predict, args=(sentence, ), result_ttl=5000
    )
    # return created job id
    return job.get_id()


@app.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):
    job = Job.fetch(job_key, connection=conn)
    if job.is_finished:
        results = job.result
        return jsonify(results)
    else:
        return "Calculating POS Tags", 202


if __name__ == '__main__':
    app.run(debug=True)
