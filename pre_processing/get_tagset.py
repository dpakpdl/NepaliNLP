import pandas as pd


def read(filename="sentences.csv"):
    data = pd.read_csv(filename, encoding="utf-8")
    data = data.fillna(method="ffill")
    print(data.tail(10))
    return data
