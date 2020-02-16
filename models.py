"""
all models
"""
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import config


def model_lgb():
    params = config.LGB_PARAM
    model = lgb.LGBMRegressor(**params)
    return model


def model_rf():
    model = RandomForestRegressor(
        n_estimators=config.n_estimator,
        max_depth = config.max_depth,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    return model

def model_KNeighborsClassifier():
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=4)

    return model


