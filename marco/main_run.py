"""
the main entrance of the project
"""
from sklearn.model_selection import KFold
from models import *
import config
import pickle
import gc
from preprocessing import load_data
from feature_engineering import extract_features
from utils import reduce_mem_usage
import lightgbm as lgb
import matplotlib.pyplot as plt


def train_model(model_name, X_train, y_train):
    kf = KFold(config.k_folds)
    cv_scores = []
    for i, (tr_idx, vl_idx) in enumerate(kf.split(X_train, y_train)):
        print('FOLD {} \n'.format(i))
        X_tr, y_tr = X_train.loc[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train.loc[vl_idx], y_train[vl_idx]

        if model_name == 'lgb':
            model = model_lgb()
            model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_vl, y_vl)], \
                      eval_metric='rmse', verbose=config.verbose, early_stopping_rounds=config.stop_rounds)
            with open('lgb_model_{}.pkl'.format(i), 'wb') as handle:
                pickle.dump(model, handle)
#code to visualize feature importance
            ax = lgb.plot_importance(model, max_num_features=40, figsize=(15,15))
            plt.show()         
            del model, X_tr, X_vl
            gc.collect()
        if model_name == 'rf':
            model = model_rf()
            model.fit(X_tr, y_tr)
            with open('rf_model_{}.pkl'.format(i), 'wb') as handle:
                pickle.dump(model, handle)
            del model, X_tr, X_vl
            gc.collect()


if __name__ == '__main__':
    df = load_data(config.DATA_PATH)
    print('data loaded with size : {}'.format(df.shape))
    X_train, y_train = extract_features(df)
    print('feature extracted')
    model_name = 'lgb'
    train_model(model_name=model_name, X_train=X_train, y_train=y_train)
    print('model training done')
    



#**************************
    
