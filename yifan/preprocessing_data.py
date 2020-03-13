import warnings

import pickle
import gc
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")



#Float range
f64_max = 1.7976931348623157e+308
f64_min =  -1.7976931348623157e+308

f32_max = 3.4028235e+38
f32_min = -3.4028235e+38

f16_max = 65500.0
f16_min = -65500.0

#Integer range
i64_max = 9223372036854775807
i64_min =  -9223372036854775808

i32_max =  2147483647
i32_min = -2147483648

i16_max = 32767
i16_min = -32768

#function for loading dataset
def load_data(path):
    df = pd.read_csv(path) # this is main table
    return df


#Function for reduce memory
def reduce_memory(train):
    for m in train.columns:
        #When it is an integer:
        if str(train[m].dtype) == 'int64':
            #Min & Max value of all columns with integer datatype
            train[m].max()
            train[m].min()
            if train[m].max() < i16_max and train[m].min() > i16_min :
                #convert column into int16
                train[m] = train[m].astype(np.int16)
            else:
                if train[m].max() < i32_max and train[m].min() > i32_min :
                    #convert column into int32
                    train[m] = train[m].astype(np.int32)
                else:
                    if train[m].max() < i64_max and train[m].min() > i64_min :
                        #convert column into int64
                        train[m] = train[m].astype(np.int64)
        
    for m in train.columns:
        #When it is a float:
        if str(train[m].dtype) == 'float64':
            #Min & Max value of all columns with float datatype
            train[m].max()
            train[m].min()
            if train[m].max() < f16_max and train[m].min() > f16_min :
                #convert column into float16
                train[m] = train[m].astype(np.float16)
            else:
                if train[m].max() < f32_max and train[m].min() > f32_min :
                    #convert column into float32
                    train[m] = train[m].astype(np.float32)
                else:
                    if train[m].max() < f64_max and train[m].min() > f64_min :
                        #convert column into float64
                        train[m] = train[m].astype(np.float64)
                        
                        
#Handle_NaN
def handle_NaN(train_df):
    # Check for Nan amount in every column
    print("start to handle NaN ...")
    nan_info = pd.DataFrame(train_df.isnull().sum()).reset_index()
    nan_info.columns = ['col', 'nan_cnt']
    nan_info.sort_values(by='nan_cnt', ascending=False, inplace=True)
    # Columns with missing values
    cols_with_missing = nan_info.loc[nan_info.nan_cnt > 0].col.values
    # Fill missing values (numbers) with the median
    for f in cols_with_missing:
        if str(train_df[f].dtype) != 'object':
            train_df[f].fillna(train_df[f].median(), inplace=True)
    # Fill missing values (objects) with Unknown
    for f in cols_with_missing:
        if str(train_df[f].dtype) == 'object':
            train_df[f].fillna('Unknown', inplace=True)
    # Check if there are still Nan values
    nan_info = pd.DataFrame(train_df.isnull().sum()).reset_index()
    nan_info.columns = ['col', 'nan_cnt']
    nan_info.sort_values(by='nan_cnt', ascending=False, inplace=True)
    
#labelencoder
def transfer_cat_2_int(train_df):
    print("Start transfer categorical values to integer ...")
    category_columns = train_df.select_dtypes(include=['category', object]).columns
    for f in category_columns:
        train_df[f] = train_df[f].astype(str)
        le = LabelEncoder()
        train_df[f] = le.fit_transform(train_df[f])