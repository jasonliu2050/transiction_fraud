"""
the main entrance of the project
"""
import warnings
from sklearn.model_selection import KFold
from models import *
import config
import pickle
import gc
import datetime
from sklearn.model_selection import train_test_split
from preprocessing_data import *
from feature_engineering import *
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def train_model(model_name, X_train, y_train):
    kf = KFold(config.k_folds)
    valid_scores = []  # validation scores
    train_scores = []  # training scores
    for i, (tr_idx, vl_idx) in enumerate(kf.split(X_train, y_train)):
        print('FOLD {} \n'.format(i))
        X_tr, y_tr = X_train.loc[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train.loc[vl_idx], y_train[vl_idx]

        if model_name == 'lgb':
            model = model_lgb()
            model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_vl, y_vl)], \
                      eval_metric='auc', verbose=config.verbose, early_stopping_rounds=config.stop_rounds)
            with open('lgb_model_{}.pkl'.format(i), 'wb') as handle:
                pickle.dump(model, handle)

            ax = lgb.plot_importance(model, max_num_features=20, figsize=(15,15))
            ax3 = lgb.plot_metric(model,figsize=(15,15))
            plt.show()
            valid_score = model.best_score_['valid_1']['auc']
            train_score = model.best_score_['training']['auc']
            valid_scores.append(valid_score)
            train_scores.append(train_score)
            del model, X_tr, X_vl
            gc.collect()
    print('Oerall scores:')
    fold_names = list(range(config.k_folds))
    metrics = pd.DataFrame({'fold': fold_names,
                          'train': train_scores,
                          'valid': valid_scores})
    print(metrics)

def train_lgb_1(model_name, X_train, y_train):
    k_fold = config.k_folds
    kf = KFold(k_fold)
    cv_scores = []

    valid_scores = []  # validation scores
    train_scores = []  # training scores

    fig, axes = plt.subplots(2, 3,figsize=(18,8))
    for i, (tr_idx, vl_idx) in enumerate(kf.split(X_train, y_train)):
        print('FOLD {} \n'.format(i))
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

        model = model_lgb()
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_vl, y_vl)], eval_metric='auc', verbose=config.verbose, early_stopping_rounds=config.stop_rounds)
        with open('lgb_model_{}.pkl'.format(i), 'wb') as handle:
            pickle.dump(model, handle)
        #code to visualize feature importance
        # axes[0][0] = lgb.plot_importance(model, max_num_features=20, figsize=(5,4))
        if i<3:
            axes[0][i] = lgb.plot_metric(model,  metric = 'auc', figsize=(4, 3),ax=axes[0][i])
        else:
            axes[1][i%3] = lgb.plot_metric(model, metric='auc', figsize=(4, 3),ax=axes[1][i%3])
            axes[1][2] = lgb.plot_importance(model, max_num_features=20, figsize=(6, 4),ax=axes[1][2])

        valid_score = model.best_score_['valid_1']['auc']
        train_score = model.best_score_['training']['auc']
        valid_scores.append(valid_score)
        train_scores.append(train_score)

        del model, X_tr, X_vl
        gc.collect()

    plt.show()
    print('Oerall scores:')
    fold_names = list(range(config.k_folds))
    metrics = pd.DataFrame({'fold': fold_names,
                          'train': train_scores,
                          'valid': valid_scores})
    print(metrics)

def train_lgb_2(model_name, X_train, y_train, X_test, y_test):
    print("start default lightGBM training ...")
    model = model_lgb_default()
    # model.fit(X_train, y_train)
    model.fit(X_train, y_train, eval_metric='auc', verbose=config.verbose)
    print(model)
    # make predictions
    expected_y = y_test
    predicted_y = model.predict(X_test)
    # summarize the fit of the model
    print(metrics.classification_report(expected_y, predicted_y))
    print("confusion_matrix:")
    print(metrics.confusion_matrix(expected_y, predicted_y))

def train_lgb_3(model_name, X_train, y_train):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = MinMaxScaler(feature_range=(0, 1))
    imputer.fit(X_train)
    train = imputer.transform(X_train)
    scaler.fit(train)
    train = scaler.transform(train)
    feat = train.copy()
    oof = np.zeros(feat.shape[0])  # out of fold predictions
    valid_scores = []  # validation scores
    train_scores = []  # training scores
    n_folds = 5
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=100)
    labels = np.array(y_train)
    i=0
    for train_indices, valid_indices in k_fold.split(feat):
        train_feat, train_labels = feat[train_indices], labels[train_indices]  # training data for the fold
        valid_feat, valid_labels = feat[valid_indices], labels[valid_indices]  # validation data for the fold
        # model = lgb.LGBMClassifier(**config.LGB_Param)
        model = model_lgb()
        model.fit(train_feat, train_labels, eval_metric='auc',
                  eval_set=[(valid_feat, valid_labels), (train_feat, train_labels)],
                  eval_names=['valid', 'train'], early_stopping_rounds=300, verbose=200)
        best_iter = model.best_iteration_
        oof[valid_indices] = model.predict_proba(valid_feat, num_iteration=best_iter)[:, 1] / k_fold.n_splits
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        valid_scores.append(valid_score)
        train_scores.append(train_score)

        with open('test_lgb_model_{}.pkl'.format(i), 'wb') as handle:
            pickle.dump(model, handle)
        i+=1
        gc.enable()
        #code to visualize feature importance
        ax = lgb.plot_importance(model, max_num_features=20, figsize=(8,6))
        ax3 = lgb.plot_metric(model,figsize=(8,6))
        plt.show()
        del model, train_feat, valid_feat
        gc.collect()

    valid_auc = roc_auc_score(labels,oof) # calculate the auc based on the test dataset labels and the out-of-fold predictions
    valid_scores.append(valid_auc) # calculate the overall validation auc score
    train_scores.append(np.mean(train_scores)) # calculate the overall average training auc score

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    metrics = pd.DataFrame({'fold': fold_names,
                          'train': train_scores,
                          'valid': valid_scores})
    print(metrics)
    
    
    
def main():
    print(datetime.datetime.now())
    print("Start to load datasets ...")
    train_tn = load_data(config.train_path)
    train_idn = load_data(config.identity_path)
    drop_selected_feature(train_tn)
    # # TEST
    #train_tn = train_tn[:10000]
    #train_idn = train_idn[:10000]
    # # TEST
    print(train_tn.shape,train_idn.shape)

    print("Drop target column, merge two tables. the new shape of main table:", train_tn.shape)
    target = train_tn['isFraud']
    train_tn.drop(['isFraud'], axis=1, inplace=True)
    train_tn = train_tn.merge(train_idn, on='TransactionID', how='left')
    train_tn.drop(['TransactionID'],axis=1, inplace=True)
    
    
    handle_NaN(train_tn)
    
    g = transaction_amt(train_tn)
    
    train_tn = handl_P_emaildomain(train_tn)
    
    train_tn = Date_trasfer(train_tn)
    
    other_feature_engineering(train_tn)
    
    transfer_cat_2_int(train_tn)
    
    reduce_memory(train_tn)

    # train lightgbm model and get result
    X = train_tn
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    model_name = 'lgb'
    train_lgb_1(model_name=model_name, X_train=X_train.to_numpy(), y_train=y_train.to_numpy())
    #train_model(model_name, X, y)
    print('model lightGBM with customised parameter done')

    # train_lgb_2(model_name=model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test )

    # train_lgb_3(model_name=model_name, X_train=X_train, y_train=y_train)

    print(datetime.datetime.now())

if __name__== "__main__":
  main()