# transiction_fraud
CEBD1260 Machine Learning Project
# Transication Fraud Detection
![matrix](./figures/frauddetection.png)

#Team member:

| Name   |
|:-------|
|Jun Liu |
|Wu      |
|Marco   |
|Saidath |

Date: 2020.03.15
-----

## Resources
This Project repository includes the following items:

- Python scripts for analysis:  
 [Project_Script](https://github.com/jasonliu2050/project/blob/master/project.py)  
- Results:  
 [Scores_Use SKlearn dataset](https://github.com/jasonliu2050/project/blob/master/figures/Scores.png)  
- runtime-instructions:  

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

The accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor. Dimensionality reduction techniques like PCA could be used prior to appplying KNN and help make the distance metric more meaningful.

Since the original dimension is quite large (784 input features), the dimensionality reduction becomes necessary. First, we extract the principal components from the original data. We do this by fitting a Principle Component Analysis (PCA) on the training set, then transforming the data using the PCA fit. We used the PCA module of the scikit-learn Python library with n_components set to differenct value to transform the dataset(use pca.explained_variance_ratio_). From the test result, I found 70 principal components can interpret approximately 90% of total information, which suffice to be representative of the information in the original dataset. We choose the first 70 principal components as the extracted features. The test result shows Accuracy and performance (fast) are much better than use all input features. ([Test Script: PCA Linear Dimensionality Reduction](https://github.com/jasonliu2050/project/blob/master/PCA_Linear_dimensionality_reduction.py))  

The following picture also show Training data size are very important to the final test accuracy result.  
([Test Script: Training sample size & Model Accuracy](https://github.com/jasonliu2050/project/blob/master/TrainingSize_Accuracy.py)) 
![matrix](./figures/PrincipalComponentAnalysis_variance.png)

### 4. Final Test Result
#### Methods Using Scikit-learn Algorithms
Picture below shows different algorithm performance. 
#### Methods Using Keras Sequential Neural Networks


## Discussion
The methods used above did solved the problem of identifying handwritten digits. These methods shows that using the current online training dataset, all  Scikit-learn algorithms: SVC, KNN, Perceptron performs very good when using the small dataset, Keras Neural Network algorithm has the best performance when using MNIST training digit dataset.  
We still need to test these algorithms performance use digit images in real life, for example, imges size may change, it may rotate in different direction, and how to handle digital image samples in dark backgroud etc. All these new issues need use Computer Vision and Pattern Recognition algorithm.(eg, OpenCV).   

## References
[Scikit-learn](https://scikit-learn.org/stable/whats_new.html#version-0-21-3) 
[Scikit-kearn Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)   
[Matplotlib3.1.1](https://matplotlib.org/3.1.1/users/whats_new.html)  
[Principal component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
