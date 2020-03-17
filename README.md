# Transication Fraud
CEBD1260 Machine Learning Project
![matrix](./figures/frauddetection.png)

Team member:
|     Name    |
|:------------|
|Jun Liu      |
|Wu Yifan     |
|Marco        |
|Saidath      |

Date: 2020.03.15
-----
## Introduction 
The financial services industry and the industries that involve financial transactions are suffering from fraud-related losses and damages. 
Using machine learning (ML) approach to fraud detection has many advantages, such as real-time processing, automatic detection of possible fraud
scenarios, and could find hidden and implicit correlations in dataset.
In this project, we use lightDBM machine learning model, after doing dataset preprocessing. feature engineering, trainning and validation, we could reach auc sconre 0.96.
### Main findings TODO
The problem we need to solve is to classify transictions. The goal is to from the transiction to determine which one is Fraud. 

### Other finding TODO

### System Pipline


#### Dataset preprocessing TODO

1. NaN handling
2. Feature drop

#### Feature Engineering TODO
3. Email domain and datatime handling
4. dataset merge and propagation 
5. Categorical values to integer convertion
6. Other feature such as addr and cards handling

#### Algorithms(models
lightGBM to solve the problem. The tasks involved are the following:

1. Model selection:
  model = lgb.LGBMClassifier(**params)

Parmeter select:

LGB_PARAM = params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'scale_pos_weight': 0.04,
    'bagging_freq': 10,
    'verbose': -1,
    'silent':-1,
    "max_depth": 10,
    "num_leaves": 128,
    "max_bin": 512,
    "n_estimators": 100000,
    'metric': 'auc',
    'random_state': seed,
    'nthread': 4
} 
2. Memory use reduction techinic

#### Methodologies:  TODO
t train/test/valid split

### 4. Final Test Result

#### Methods Using Scikit-learn Algorithms
Picture below shows different algorithm performance. 
#### Methods 


## Statement of contributions
Jun Liu      github management, prototyping and integration of project, documention

Wu  Yifan    Handle Feature 'TransactionAmt', 'Datetime', Feature Drop, Documention  

Marco        
Saidath      


-----
# Data Preprocessing 

- This part bascially we need handle the NAN values in our dataset 

- Transfer 'object' values into digital data

- Reduce Memory

-----

# Handle NAN Value

- NAN Value has crucial part in data preprocessing. what we did is fill in the float and integer columns with their median, fill in 'unknow' into the 'object' columns.

- Since if we fill in average values may has a chance to get very unbalanced dataset,therefore median values should be more Suitable

- Here is the code we use:

![Handle_nan](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/handle%20Nan.png)

-----

# Reduce Memory and Transfer values

- Since computer can only read the digital values that we have to transfer all the 'object' values into the digital values.

- Memory Reduce can help us save the time when we run our model, it can improve our effiency of work.

-----



# Feature Engineering 

- This part we are going to explain the Feature Engineering that we did for our project and the purpose of doing it.
  
Bascially there are 4 big part for the whole feature enginnering that we are going to explain about it.
  
-----


### ['P_emaildomain'] Feature 
- Handle the Nonsense value and transfer them in some value in order to make computer to read it:

- The red circle is the one that we need to take care of it:

![P_email](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/unique_of_email.png)

- Therefore we would like to replace those value then when we will get less pressure when we train our model.

- Also we create an new feature called ['Region_emaildomain']:

- what we did is transfer all the information that we got in ['P_emaildomain'] and then to get all the exact country where those email    sent, which we think the location of those email is going to help our model a lot since each country must have their characteristic. Here is the code below: 

![P_email](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/email_code.png)

-----


## ['TransactionAmt'] Feature

- This feature bascilly has lots of outlier that we need to handle since we displot it out and find out this one is long_tail type: 

![long_tail](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/long_amt.png)

- we think that if this feature has too many outliers it might make our model unsteadable, therefore we decide to drop the outlier values in this feature. 

- Here is the code that we use:

![code_for drop](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/code_for_drop_outlier.png)

- we decide to drop the value which is greater than the 99 percent of the values also the values which is less than 1 percent. Here is the Figure after drop: 

![drop_outlier](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/drop_outlier.png)

-----


## ['TransactionDT'] Feature

- This feature contains all the transaction date time which is an object type in the oringal dataframe. Therefore we think we need to transfer them to the dateobject in order to let our computer read it and run.

- More specify we need to add some columns like: 'df['day']', ' df['year']' etc, which is some importance features that transfer it from the ['TransactionDT'].

- Here is the code that we use:

![code_date](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/code_date_transfer.png)


-----

### ['Card1'], ['Card2'] Features

- .There are some importance features such as ['Card1'], ['Card2'] that has huge amounts of values which is really complex and messed. we think these feature need to be handle it into the category type value. 

- Category values are going to relase some pressure when we are going to train our model. Therefore we find those features and set up some bins(threshold). in this way, the data is more clean and orangize. 

- Also we think only some importance features that need it to be handle in category type since they can affect the model a lot.

- Here is the code that we use:

![other_feature](https://github.com/jasonliu2050/transiction_fraud/blob/master/figures/feature_and_preprocess/other_feature_code.png)


## References
[Scikit-learn](https://scikit-learn.org/stable/whats_new.html#version-0-21-3) 
[Scikit-kearn Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)   
[Matplotlib3.1.1](https://matplotlib.org/3.1.1/users/whats_new.html)  
[Principal component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
[pandas](https://pandas.pydata.org/docs/)
[Seaborn](https://seaborn.pydata.org/)
