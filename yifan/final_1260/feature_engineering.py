import warnings

import datetime
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


# Function to clean the names
def assign_region(email_addr):
    REGION = {
        ".jp": "Japan",
        ".fr": "French",
        ".uk": "UK",
        ".mx": "Mexico",
        ".de": "German",
        ".es": "Spain",
        ".com": "Global",
        ".net": "Global",
        "Unknown": "Unknown"
        }
    for key in REGION.keys():
        if email_addr.find(key) != -1:
            return REGION[key]
        
# do the email columns
def handl_P_emaildomain(train_df):
    #create a new column
    train_df = train_df.assign(Region_emaildomain = train_df['P_emaildomain'])
    # process P_emaildomain column
    train_df.loc[train_df['P_emaildomain'] == 'gmail', 'Region_emaildomain'] = 'gmail.com'
    train_df.loc[train_df['P_emaildomain'] == 'gmail', 'P_emaildomain'] = 'gmail.com'
    train_df['Region_emaildomain'] = train_df['Region_emaildomain'].apply(assign_region)
    
    print(train_df['Region_emaildomain'].head())
    return train_df

#handle Non sense feature
def drop_selected_feature(train_df):
    features = ['V322', 'V323', 'V326', 'V330', 'V331', 'V333', 'V334', 'V336', 'V337', 'V339',
     'V2', 'V4', 'V6', 'V8', 'V11', 'V324', 'V329', 'V300', 'V10', 'V16', 'V17', 'V22', 'V27', 'V29', 'V31', 'V33',
     'V35', 'V39', 'V42', 'V48', 'V51', 'V57', 'V59', 'V63', 'V69', 'V71', 'V73', 'V80', 'V84', 'V90', 'V92', 'V96', 'V103', 'V105', 'V127',
     'V139', 'V148', 'V153', 'V155', 'V150', 'V178', 'V182', 'V192', 'V204', 'V212', 'V219', 'V224', 'V233', 'V248',
     'V221', 'V238', 'V250', 'V255', 'V272', 'V295', 'V299', 'V308', 'V318']
    for f in features:
        train_df.drop([f], axis=1,inplace=True)
        
#handle transaction_amt
def transaction_amt(train_txn):
    g = train_txn['TransactionAmt']
    #y is the value that is  1 percent and x is the value that is 99 percent
    y = g.quantile(q=0.01)
    x = g.quantile(q=0.999)
    list1 =[] #make a list that we need to drop 
    for i in g.index:
        if g[i] < y or g[i] > x:
            list1.append(i)
            
    g = g.drop(list1) # drop the outlier values
      
    N12 = np.log(g)
    
    train_txn['N12'] = N12
    
    return g


#handle date features
def Date_trasfer(df):
    df['TransactionDT'] = pd.to_datetime(df['TransactionDT'], unit='s', origin=pd.Timestamp('2017-11-30'))
    df['day'] =df['TransactionDT'].dt.day
    df['year'] = df['TransactionDT'].dt.year
    df['hour'] = df['TransactionDT'].dt.hour
    df['min'] = df['TransactionDT'].dt.minute
    df['second'] = df['TransactionDT'].dt.second
    df['doy'] = df['TransactionDT'].dt.dayofyear
    df.drop(['TransactionDT'],axis=1,inplace=True)
    return df


#handle other features
def other_feature_engineering(train_txn):
    
    cut_labels = ['1', '2', '3', '4']
    cut_bins = [0, 200, 300, 400, train_txn['addr1'].max()]
    train_txn['addr1'] = pd.cut(train_txn['addr1'], bins=cut_bins, labels=cut_labels)

    cut_labels = ['1', '2', '3', '4', '5']
    cut_bins = [0, 20, 40, 60, 80, train_txn['addr2'].max()]
    train_txn['addr2'] = pd.cut(train_txn['addr2'], bins=cut_bins, labels=cut_labels)

    cut_labels = ['1', '2', '3', '4', '5', '6']
    cut_bins = [0, 4001, 7001, 10000, 13000, 16000, train_txn['card1'].max()]
    train_txn['card1'] = pd.cut(train_txn['card1'], bins=cut_bins, labels=cut_labels)

    cut_labels = ['1', '2', '3', '4', '5']
    cut_bins = [0, 200, 300, 400, 500, train_txn['card2'].max()]
    train_txn['card2'] = pd.cut(train_txn['card2'], bins=cut_bins, labels=cut_labels)
    
    