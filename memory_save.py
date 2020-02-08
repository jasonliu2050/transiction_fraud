import os
import gc   # for gabage collection
import seaborn as sns  # data visualization lib
import matplotlib.pyplot as plt
import glob


# some customized function for plotting data
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


file_name='E:\\machine_learning\\transaction_fraud\\train_transaction.csv'
train_df = load_data(file_name)
file_name1 = 'E:\\machine_learning\\transaction_fraud\\train_identity.csv'
identity_df = load_data(file_name)

# e.g., I want to check how much momery session_df takes
mem_use = identity_df.memory_usage().sum() / 1024**2  # convert bytes to MB by dividing 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mem_use))


features = [f for f in train_df.columns.values if f not in ['isFraud']]
features_int64 = []
changed = 0
for f in features:
    if str(train_df[f].dtype) in ['float64']:
        if train_df[f].max() < 65500 and train_df[f].min()> -65500:
            train_df[f] = train_df[f].astype(np.float16)
            changed +=1


print('changed columns = ', changed)
mem_use = train_df.memory_usage().sum() / 1024**2  # convert bytes to MB by dividing 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mem_use))
yy
