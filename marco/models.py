"""
all models
"""
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import config


def model_lgb():
    params = config.LGB_PARAM
#   model = lgb.LGBMRegressor(**params)
    model = lgb.LGBMClassifier(**params)
    return model


def model_rf():
    model = RandomForestRegressor(
        n_estimators=config.n_estimator,
        max_depth = config.max_depth,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    return model

def your_model():
    # your model here


    pass


