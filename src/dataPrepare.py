import pandas as pd
import numpy as np

# df_normal = pd.read_csv('data/ptbdb_normal.csv', header=None, nrows=4000)
# df_abnormal = pd.read_csv('data/ptbdb_abnormal.csv', header=None, nrows=4000)
df_normal = pd.read_csv('data/ptbdb_normal.csv', header=None)
df_abnormal = pd.read_csv('data/ptbdb_abnormal.csv', header=None)

# ''' add the class column '''
#
# df_normal[df_normal.shape[1] + 1] = 0  # class 0 for normal values
# df_abnormal[df_abnormal.shape[1] + 1] = 1  # class 1 for abnormal values

# ''' concat the dataframes and shuffle '''
#
# df = pd.concat([df_normal, df_abnormal]).sample(frac=1)

''' concat the dataframes without shuffle'''

df = pd.concat([df_normal, df_abnormal])

''' split dataframe into train and test by 80/20 '''

msk = np.random.rand(len(df)) < 0.8

train_data = df[msk]
test_data = df[~msk]

''' write data to csv files'''

train_data.to_csv('data/ptbdb_train.csv', header=False, index=False)
test_data.to_csv('data/ptbdb_test.csv', header=False, index=False)
df.to_csv('data/ptbdb.csv', header=False, index=False)