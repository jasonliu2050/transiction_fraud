"""
all functions related to preprocessing
"""
import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path + 'train_transaction.csv', nrows=50000)#, compression='zip', usecols=usecols)
    return df
