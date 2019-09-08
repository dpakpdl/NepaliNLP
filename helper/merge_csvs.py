import os

import pandas as pd
from utility import get_training_data_path, get_testing_data_path, get_data_path

df1 = pd.read_csv(get_training_data_path())
bottom = df1.tail(1)
print(bottom.sentence)

df2 = pd.read_csv(get_testing_data_path())
df2.sentence += int(bottom.sentence)

print(df2.tail(1))

df = df1.append(df2)
merge_data_path = os.path.join(get_data_path(), "UrduNepaliEnglish", "merged.csv")
df.to_csv(merge_data_path, sep=',', index=None, header=True)
