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
train_transc = pd.read_csv(file_transc)
print(train_transc.shape)
train_transc.head()


# In[4]:


#IDENTITY TABLE
train_idn = pd.read_csv(file_idn)
print(train_idn.shape)
train_idn.head()


# In[ ]:





# In[5]:


#Filling Transaction table


# In[6]:


#Check for Nan amount in every column
nan_info = pd.DataFrame(train_transc.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
nan_info


# In[7]:


#Columns with missing values
cols_with_missing = nan_info.loc[nan_info.nan_cnt>0].col.values
cols_with_missing


# In[8]:


#Fill missing values (numbers) with the median
for f in cols_with_missing:
    if str(train_transc[f].dtype) != 'object':
        train_transc[f].fillna(train_transc[f].median(),inplace=True)
        


# In[9]:


#Fill missing values (objects) with Unknown
for f in cols_with_missing:
    if str(train_transc[f].dtype) == 'object':
        train_transc[f].fillna('Unknown',inplace=True)
        


# In[10]:


#Check if there are still Nan values 
nan_info = pd.DataFrame(train_transc.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
nan_info


# In[ ]:





# In[11]:


#Filling Identity Table


# In[12]:


#Check for Nan amount in every column
nan_info = pd.DataFrame(train_idn.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
nan_info


# In[13]:


#Columns with missing values
cols_with_missing = nan_info.loc[nan_info.nan_cnt>0].col.values
cols_with_missing


# In[14]:


#Fill missing values (numbers) with the median
for f in cols_with_missing:
    if str(train_idn[f].dtype) != 'object':
        train_idn[f].fillna(train_idn[f].median(),inplace=True)
        


# In[15]:


#Fill missing values (objects) with Unknown
for f in cols_with_missing:
    if str(train_idn[f].dtype) == 'object':
        train_idn[f].fillna('Unknown',inplace=True)
        


# In[16]:


#Check if there are still Nan values
nan_info = pd.DataFrame(train_idn.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
nan_info


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




