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


# In[ ]:





# In[2]:


#Getting the file path
DATA_PATH = '/Users/TygaBRii/Downloads/ML/'
file_transc = DATA_PATH+'train_transaction.csv'
file_idn = DATA_PATH+'train_identity.csv'
print(file_transc,'\n', file_idn)


# In[3]:


#TRANSACTION TABLE
train_txn = pd.read_csv(file_transc)
print(train_txn.shape)
train_txn.head()


# In[4]:


#IDENTITY TABLE
train_idn = pd.read_csv(file_idn)
print(train_idn.shape)
train_idn.head()


# In[ ]:





# # Handling Addr1 & 2
# Creating a range for both adress reducing categories
# 

# In[5]:


train_txn[['addr1', 'addr2']].head()


# In[6]:


#   ADDR1

print('Unique vaues : ', train_txn.addr1.nunique())
print('Max value :', train_txn['addr1'].max())
print('Min value :', train_txn['addr1'].min())


# In[7]:


#   ADDR2

print('Unique vaues : ', train_txn.addr2.unique())
print('Max value :', train_txn['addr2'].max())
print('Min value :', train_txn['addr2'].min())


# ### Assiging the range to adrr1 & 2

# In[8]:


# Addr1

cut_labels = ['1', '2', '3', '4']
cut_bins = [0, 200, 300, 400, train_txn['addr1'].max()]
train_txn['addr1'] = pd.cut(train_txn['addr1'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['addr1'].unique())


# In[9]:


#Addr2

## range : 0-20 is 1, 21-40 is 2, 41-60 is 3, 61-80 is 4, 81-max is 5
## Label : 1-5

cut_labels = ['1', '2', '3', '4', '5']
cut_bins = [0, 20, 40, 60, 80, train_txn['addr2'].max()]
train_txn['addr2'] = pd.cut(train_txn['addr2'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['addr2'].unique())


# # Handling card1-6
# 
# card4 & card 6 are already grouped into good categories.
# 
# A range was created for each column and each range was grouped into one category.
# For ex : range 1 to 100 wil be category 1 and 101 to 200 will be category 2
#                         

# In[10]:


train_txn[['card1', 'card2', 'card3', 'card4', 'card5', 'card6']].head()


# In[ ]:





# In[11]:


#   CARD1

print('Unique values:', train_txn.card1.unique())
print('Number of unique categories : ', train_txn.card1.nunique())
    
print('Max value :', train_txn['card1'].max())
print('Min value :', train_txn['card1'].min())


# In[12]:


# CARD1

cut_labels = ['1', '2', '3', '4', '5', '6']
cut_bins = [0, 4001, 7001, 10000, 13000, 16000, train_txn['card1'].max()]
train_txn['card1'] = pd.cut(train_txn['card1'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['card1'].unique())


# In[13]:


#   CARD2

print('Number of unique categories : ', train_txn.card2.nunique())
    
print('Max value :', train_txn['card2'].max())
print('Min value :', train_txn['card2'].min())


# In[14]:


# CARD2

cut_labels = ['1', '2', '3', '4', '5']
cut_bins = [0, 200, 300, 400, 500, train_txn['card2'].max()]
train_txn['card2'] = pd.cut(train_txn['card2'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['card2'].unique())


# In[15]:


#   CARD3

print('Number of unique categories : ', train_txn.card3.nunique())
    
print('Max value :', train_txn['card3'].max())
print('Min value :', train_txn['card3'].min())


# In[16]:


# CARD3

cut_labels = ['1', '2', '3']
cut_bins = [0, 150, 200, train_txn['card3'].max()]
train_txn['card3'] = pd.cut(train_txn['card3'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['card3'].unique())


# In[17]:


# CARD4
print('Unique values:', train_txn.card4.unique())


# In[18]:


# CARD5

print('Number of unique categories : ', train_txn.card5.nunique())
    
print('Max value :', train_txn['card5'].max())
print('Min value :', train_txn['card5'].min())


# In[19]:


# CARD5

cut_labels = ['1', '2', '3']
cut_bins = [0, 150, 200, train_txn['card5'].max()]
train_txn['card5'] = pd.cut(train_txn['card5'], bins=cut_bins, labels=cut_labels)

print('Unique values:', train_txn['card3'].unique())


# In[20]:


#   CARD6
print('Unique values:', train_txn.card6.unique())


# In[ ]:





# In[ ]:




