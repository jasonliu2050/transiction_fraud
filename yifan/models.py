"""
all models
"""
import lightgbm as lgb

import config


def model_lgb():
    params = config.LGB_PARAM
    model = lgb.LGBMClassifier(**params)
    return model

def model_lgb_default():
    model = lgb.LGBMClassifier()
    return model
