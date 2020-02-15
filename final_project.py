
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

train_path='h:\\machine_learning\\transaction_fraud\\train_transaction.csv'
identity_path = 'h:\\machine_learning\\transaction_fraud\\train_identity.csv'

def load_data(path):
    df = pd.read_csv(path) # this is main table
    return df

def drop_corr_column(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features
    df.drop(df[to_drop], axis=1, inplace=True)


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
# Function to clean the names
def assign_region(email_addr):
    for key in REGION.keys():
        if email_addr.find(key) != -1:
            return REGION[key]

def main():
    print("Start to load datasets ...")

    train_df = load_data(train_path)
    identity_df = load_data(identity_path)
    print(train_df.shape,identity_df.shape)
    target = train_df['isFraud']
    train_df.drop(['isFraud'], axis=1, inplace=True)

    train_df = train_df.merge(identity_df, on='TransactionID', how='left')
    print("Drop target column, merge two tables. the new shape of main table:", train_df.shape)

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

    print("Start transfer categorical values to integer ...")
    category_columns = train_df.select_dtypes(include=['category', object]).columns
    for f in category_columns:
        train_df[f] = train_df[f].astype(str)
        le = LabelEncoder()
        train_df[f] = le.fit_transform(train_df[f])

    print("Start drop corr columns ...")
    drop_corr_column(train_df)
    print(train_df.columns.values)  # check all columns names
    train_df.info()

    print("Start reduce memory ...")
    reduce_memory(train_df)

    print("Start to train ....")
    train = train_df.to_numpy()
    target = target.to_numpy()

    y = target
    X = train
    # Scaling data (KNeighbors methods do not scale automatically!)
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35)

    i = 30
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    f1_scores = f1_score(y_test, y_predicted, average="macro")
    error_rate = np.mean(y_predicted != y_test)

    print("\n\nTrain result: ")
    print(f1_scores, error_rate)

if __name__== "__main__":
  main()
