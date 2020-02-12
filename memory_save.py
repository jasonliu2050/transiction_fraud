import numpy as np   # import numpy
import pandas as pd  # import pandas
import os
import gc   # for gabage collection
import seaborn as sns  # data visualization lib
import matplotlib.pyplot as plt
import glob

train='h:\\machine_learning\\transaction_fraud\\train_transaction.csv'
identity = 'h:\\machine_learning\\transaction_fraud\\train_identity.csv'

def plot_corr(df, cols):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    plt.show()


def plot_count(df, col, fsize, rotation=None, fillna=True):
    fig = plt.figure(figsize=fsize)
    if fillna:
        value_count = df[col].fillna('unknown').value_counts()
    sns.barplot(value_count.index, value_count.values)
    plt.xticks(rotation=rotation)
    plt.title('value counts for {}'.format(col))
    plt.show()

def load_data(path):
    df = pd.read_csv(path) # this is main table
    return df
# print data type for each feature
def printAllDataType(features):
    for f in features:
        try:
            print(" {}: {}".format(f,train_df[f].dtype))
        except AttributeError:
            pass

def reduce_memory(dataframe):
    features = [f for f in dataframe.columns.values if f not in ['isFraud']]
    for f in features:
        try:
            if dataframe[f].dtype == np.float64:
                if dataframe[f].nunique() == 2:
                    if 'T' in list(dataframe[f].unique()) or 'F' in list(dataframe[f].unique()):
                        dataframe[f].fillna('F')
                        dataframe[f] = (dataframe[f] == 'T').astype(np.bool)
                    else:
                        dataframe[f].fillna(0)
                        dataframe[f] = dataframe[f].astype(np.bool)
                elif dataframe[f].max() < np.finfo(np.float16).max and dataframe[f].min()> np.finfo(np.float16).min:
                    dataframe[f] = dataframe[f].astype(np.float16)
                elif dataframe[f].max() < np.finfo(np.float32).max and dataframe[f].min()> np.finfo(np.float32).min:
                    dataframe[f] = dataframe[f].astype(np.float32)
            elif dataframe[f].dtype == np.int64:
                if dataframe[f].max() <= 255 and dataframe[f].min()>=0 :
                    dataframe[f] = dataframe[f].astype(np.int8)
                if dataframe[f].max() < 32767 and dataframe[f].min()>= - 32767 :
                    dataframe[f] = dataframe[f].astype(np.int16)
                elif dataframe[f].max() < 2147483647 and dataframe[f].min()>= - 2147483647 :
                    dataframe[f] = dataframe[f].astype(np.int32)
        except AttributeError:
            pass
    mem_use = dataframe.memory_usage().sum() / 1024**2  # convert bytes to MB by dividing 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(mem_use))

def main():
    train_df = load_data(train)
    identity_df = load_data(identity)

    reduce_memory(train_df)
    reduce_memory(identity_df)

if __name__== "__main__":
  main()
