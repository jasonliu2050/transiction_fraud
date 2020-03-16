# Transication Fraud
CEBD1260 Machine Learning Project
![matrix](./figures/frauddetection.png)

Team member:
|     Name    |
|:------------|
|Jun Liu      |
|Wu           |
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
#### Methods Using Keras Sequential Neural Networks


## Statement of contributions
Jun Liu      github management, prototyping and integration of project, documention
Wu           
Marco        
Saidath      

## References
[Scikit-learn](https://scikit-learn.org/stable/whats_new.html#version-0-21-3) 
[Scikit-kearn Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)   
[Matplotlib3.1.1](https://matplotlib.org/3.1.1/users/whats_new.html)  
[Principal component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
