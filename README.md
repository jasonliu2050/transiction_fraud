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
In this project, we use lightDBM machine learning model, after doing dataset preprocessing. feature engineering, trainning and validation, we could reach 
auc sconre 0.96.
### Main findings TODO
The problem we need to solve is to classify handwritten digits. The goal is to take an image of a handwritten digit and determine what that digit is. The digits range from 0 through 9. We could apply machine learning algorithms to solve our problem. Using simple handwriting digit datasets provided by Scikit-learn, we achieved from 97% F1-score with Nearest Neighbor (KNN) classifier, to achieving 99% F1 score with Support Vector Classifier(SVC). The scope of this article also include comparing the different classifiers, using dataset from the famous MNIST (Modified National Institute of Standards and Technology), and try to achieve higher performance by choosing parameters along with dataset preprocessing technique.
### Other finding TODO

### Methods Using LightGBM

lightGBM to solve the problem. The tasks involved are the following:

1. Load and Explore the Digit Dataset
2. Simple visualization and classification of the digits dataset
3. Dataset Preprocessing 
4. Train a classifier, and test its accuracy

#### 3. Dataset preprocessing
I give a example for data preprocessing when use Nearest Neighbor (KNN) classifier.


### 4. Final Test Result
#### Methods Using Scikit-learn Algorithms
Picture below shows different algorithm performance. 
#### Methods Using Keras Sequential Neural Networks


## Discussion
  

## References
[Scikit-learn](https://scikit-learn.org/stable/whats_new.html#version-0-21-3) 
[Scikit-kearn Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)   
[Matplotlib3.1.1](https://matplotlib.org/3.1.1/users/whats_new.html)  
[Principal component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
