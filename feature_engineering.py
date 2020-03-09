"""
all functions about feature engineering or analysis
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder

def drop_corr_column(df):
    print("Start drop corr columns ...")
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features
    df.drop(df[to_drop], axis=1, inplace=True)
    #print(df.info())

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
        ".net": "Global"
    }
    for key in REGION.keys():
        if email_addr.find(key) != -1:
            return REGION[key]


def handl_P_emaildomain(train_df):
    print("start to handle  P_emaildomain...")
    # process NaN value
    train_df['P_emaildomain'].fillna('TBD', inplace=True)
    #create a new column
    train_df = train_df.assign(Region_emaildomain=train_df['P_emaildomain'])
    # process P_emaildomain column
    train_df.loc[train_df['P_emaildomain'] == 'TBD', 'Region_emaildomain'] = 'Global'
    train_df.loc[train_df['P_emaildomain'] == 'yahoo', 'P_emaildomain'] = 'yahoo.com'
    train_df.loc[train_df['P_emaildomain'] == 'gmail', 'P_emaildomain'] = 'gmail.com'
    train_df['Region_emaildomain'] = train_df['Region_emaildomain'].apply(assign_region)
    return train_df

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
    #print(nan_info)

def transfer_cat_2_int(train_df):
    print("Start transfer categorical values to integer ...")
    category_columns = train_df.select_dtypes(include=['category', object]).columns
    for f in category_columns:
        train_df[f] = train_df[f].astype(str)
        le = LabelEncoder()
        train_df[f] = le.fit_transform(train_df[f])

def other_feature_engineering(train_txn):
    # train_txn['TransactionDT']= datetime.datetime.fromtimestamp(train_txn['TransactionDT'][0]).strftime("%A, %B %d, %Y %I:%M:%S")
    # train_txn['year'] = train_txn.TransactionDT.dt.year
    # train_txn['month'] = train_txn.TransactionDT.dt.month
    # train_txn['day'] = train_txn.TransactionDT.dt.day
    # train_txn['hour'] = train_txn.TransactionDT.dt.hour
    # train_txn['minute'] = train_txn.TransactionDT.dt.minute

    # cut_labels = ['1', '2', '3', '4']
    # cut_bins = [0, 200, 300, 400, train_txn['addr1'].max()]
    # train_txn['addr1'] = pd.cut(train_txn['addr1'], bins=cut_bins, labels=cut_labels)
    #
    # cut_labels = ['1', '2', '3', '4', '5']
    # cut_bins = [0, 20, 40, 60, 80, train_txn['addr2'].max()]
    # train_txn['addr2'] = pd.cut(train_txn['addr2'], bins=cut_bins, labels=cut_labels)
    #
    # cut_labels = ['1', '2', '3', '4', '5', '6']
    # cut_bins = [0, 4001, 7001, 10000, 13000, 16000, train_txn['card1'].max()]
    # train_txn['card1'] = pd.cut(train_txn['card1'], bins=cut_bins, labels=cut_labels)
    #
    # cut_labels = ['1', '2', '3', '4', '5']
    # cut_bins = [0, 200, 300, 400, 500, train_txn['card2'].max()]
    # train_txn['card2'] = pd.cut(train_txn['card2'], bins=cut_bins, labels=cut_labels)

    return train_txn

def drop_selected_feature(train_txn):
    features = ['V322', 'V323', 'V326', 'V330', 'V331', 'V333', 'V334', 'V336', 'V337', 'V339',
     'V2', 'V4', 'V6', 'V8', 'V11', 'V324', 'V329', 'V300', 'V10', 'V16', 'V17', 'V22', 'V27', 'V29', 'V31', 'V33',
     'V35', 'V39', 'V42', 'V48', 'V51', 'V57', 'V59', 'V63', 'V69', 'V71', 'V73', 'V80', 'V84', 'V90', 'V92', 'V96', 'V103', 'V105', 'V127',
     'V139', 'V148', 'V153', 'V155', 'V150', 'V178', 'V182', 'V192', 'V204', 'V212', 'V219', 'V224', 'V233', 'V248',
     'V221', 'V238', 'V250', 'V255', 'V272', 'V295', 'V299', 'V308', 'V318']
    for f in features:
        train_txn.drop([f], axis=1,inplace=True)