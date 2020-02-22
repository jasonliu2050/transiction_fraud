#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Getting the file path
DATA_PATH = '/Users/TygaBRii/Downloads/ML/'
file_transc = DATA_PATH+'train_transaction.csv'
file_idn = DATA_PATH+'train_identity.csv'
print(file_transc,'\n', file_idn)


# In[ ]:





# In[24]:


#TRANSACTION TABLE
train_txn = pd.read_csv(file_transc)
print(train_txn.shape)


# In[25]:


#IDENTITY TABLE
train_idn = pd.read_csv(file_idn)
print(train_idn.shape)


# In[ ]:





# In[5]:


def get_nan_dic(df):
    """
    get NaN dictionary
    return: a dictionary with #of null values as key and feature names as value
    """
    nulls = df.isna()
    nan_dic = {}
    for f in df.columns:
        c=nulls[f].sum()
        nan_dic[c]=[]
    for f in df.columns:
        c = nulls[f].sum()
        nan_dic[c].append(f)
    return nan_dic


# In[6]:


nan_dic = get_nan_dic(train_txn)


# In[7]:


print(nan_dic)


# In[8]:


for key, value in nan_dic.items() :
    print(key)


# In[9]:


#To plot correlation matrix as heatmap
def plot_corr(df,cols):
    fig = plt.figure(figsize=(12,8))
    sns.heatmap(df[cols].corr(),cmap='RdBu_r', annot=True, center=0.0)
    plt.show() 


# In[10]:


Vs = nan_dic[279287]
Vs


# In[11]:


plot_corr(train_txn,Vs)


# In[12]:


Vs = nan_dic[525823] + nan_dic[528588] + nan_dic[528353] + nan_dic[89113] 
Vs    


# In[13]:


plot_corr(train_txn,Vs)


# Columns M1 to M7 are object type column and cannot be plotted using the heatmap or using direct correlation
# 
# nan_dic[271100, 281444, 350482, 169360, 346265]
# 
# The object categories have to be converted to numerical values

# In[14]:


##Label Encoder


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


def transfer_cat_2_int(train_df):
    print("Start transfer categorical values to integer ...")
    category_columns = train_df.select_dtypes(include=['category', object]).columns
    for f in category_columns:
        train_df[f] = train_df[f].astype(str)
        le = LabelEncoder()
        train_df[f] = le.fit_transform(train_df[f])


# In[17]:


transfer_cat_2_int(train_txn)


# In[18]:


Vs = nan_dic[271100] + nan_dic[281444] + nan_dic[350482] + nan_dic[169360] + nan_dic[346265]
Vs


# In[19]:


plot_corr(train_txn,Vs)


# In[ ]:





# In[22]:


for column in train_txn.columns:
    corr = train_txn['isFraud'].corr(train_txn[column])
    print(column, corr)


# In[ ]:





# CONCLUSION :
# I compared them against each other and also our target to find out which one should be dropped.
# 
# For the Vs :
# Corr    - Keep - Drop
# V2 & V3 -  V3  - V2
# V4 & V5 -  V5  - V4
# V6 & V7 -  V7  - V6
# V8 & V9 -  V9  - V8
# V10& V11-  V10 - V11
# 
# For the Ds :
# None of them have a high correlation so it would be safe to keep them all.
# 
# For the Ms:
# Correlated  - Keep
# M1, M2 & M3 - M1 **More correlated to isFraud column
# M4 & M5     - M4
# M1 & M7     - M1 **already keeping M1
# 
# 
# 

# In[ ]:




