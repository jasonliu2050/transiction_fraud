"""
all functions about feature engineering or analysis
"""
import pandas as pd
import numpy as np
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
    print(df.info())

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
    print(nan_info)

def transfer_cat_2_int(train_df):
    print("Start transfer categorical values to integer ...")
    category_columns = train_df.select_dtypes(include=['category', object]).columns
    for f in category_columns:
        train_df[f] = train_df[f].astype(str)
        le = LabelEncoder()
        train_df[f] = le.fit_transform(train_df[f])
