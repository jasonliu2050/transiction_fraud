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
from utils import *
from feature_engineering import *

warnings.filterwarnings("ignore")

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
            del model, X_tr, X_vl
            gc.collect()
        if model_name == 'rf':
            model = model_rf()
            model.fit(X_tr, y_tr)
            with open('rf_model_{}.pkl'.format(i), 'wb') as handle:
                pickle.dump(model, handle)
            del model, X_tr, X_vl
            gc.collect()



def train_KNeighborsClassifier(X_train, y_train):
    print("Start to train KNeighborsClassifier ....")
    train = X_train.to_numpy()
    target = y_train.to_numpy()

    y = target
    X = train
    # Scaling data (KNeighbors methods do not scale automatically!)
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35)

    knn = KNeighborsClassifier(n_neighbors=10, n_jobs=4)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    f1_scores = f1_score(y_test, y_predicted, average="macro")
    error_rate = np.mean(y_predicted != y_test)
    return f1_scores, error_rate

def main():
    print(datetime.datetime.now())
    print("Start to load datasets ...")
    train_df = load_data(config.train_path)
    identity_df = load_data(config.identity_path)

    # # TEST
    # train_df = train_df[:10000]
    # identity_df = identity_df[:10000]
    # # TEST
    print(train_df.shape,identity_df.shape)

    print("Drop target column, merge two tables. the new shape of main table:", train_df.shape)
    target = train_df['isFraud']
    train_df.drop(['isFraud'], axis=1, inplace=True)
    train_df = train_df.merge(identity_df, on='TransactionID', how='left')

    handl_P_emaildomain(train_df)

    handle_NaN(train_df)

    transfer_cat_2_int(train_df)

    drop_corr_column(train_df)

    reduce_mem_usage(train_df)

    # train knn model and get result
    # f1_scores, error_rate = train_KNeighborsClassifier(train_df, target)
    # print(f1_scores, error_rate)

    # train lgb model and get result
    X = train_df   #.to_numpy()
    y = target     #.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1000)
    model_name = 'lgb'
    train_model(model_name=model_name, X_train=X_train, y_train=y_train)
    print('model lightGBM training done')


    print(datetime.datetime.now())

if __name__== "__main__":
  main()
