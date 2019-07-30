import pandas as pd


def read():
    data = pd.read_csv("sentences.csv", encoding="utf-8")
    data = data.fillna(method="ffill")
    print(data.tail(10))
    return data
