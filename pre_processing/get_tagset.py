import os

import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def read():
    file_path = os.path.join(THIS_DIR, 'sentences.csv')
    data = pd.read_csv(file_path, encoding="utf-8")
    data = data.fillna(method="ffill")
    print(data.tail(10))
    return data
