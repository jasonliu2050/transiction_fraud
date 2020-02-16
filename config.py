"""
all parameters and configurations
"""
train_path='h:\\machine_learning\\transaction_fraud\\train_transaction.csv'
identity_path = 'h:\\machine_learning\\transaction_fraud\\train_identity.csv'

n_estimator = 100
max_depth = 10
seed = 10
n_jobs = -1
stop_rounds = 1000
k_folds = 5
verbose = 500

LGB_PARAM = params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': -1,
    'silent':-1,
    "max_depth": 10,
    "num_leaves": 128,
    "max_bin": 512,
    "n_estimators": 100000
}
