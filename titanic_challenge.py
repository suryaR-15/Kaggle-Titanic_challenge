# -*- coding: utf-8 -*-
"""
@author: surya

Kaggle titanic challenge: https://www.kaggle.com/c/titanic/overview

XGB Classifier has been used after feature engineering
Cross-validation approach used; best model used to predict labels on test data

Kaggle accuracy score achieved using this version of the code - 0.80382
"""


import pandas as pd
import numpy as np

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from feature_engineering import feature_eng

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

# =============================================================================
# Read in and look at train and test data - uncomment as required
# =============================================================================
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
# print(train_data.columns)
# print(train_data.describe())
# print(train_data.head())
# print(train_data.shape)

# =============================================================================
# Feature engineering defined as a separate function - specific to this
# dataset
# =============================================================================
X_train, y_train = feature_eng(train_data, 'training')
X_test, y_test = feature_eng(test_data, 'test')

# =============================================================================
# Define the helper class named 'model' that defines fit, predict etc. for
# all classifiers so they do not have to be repeated in the code
# =============================================================================
seed = 0
nfolds = 5


class model(object):
    def __init__(self, mdl, params=None):
        self.mdl = mdl(**params)

    def fit(self, x, y):
        return self.mdl.fit(x, y)

    def predict(self, x):
        return self.mdl.predict(x)


# =============================================================================
# K-fold raining. Predictions directly made on the Kaggle test data
# =============================================================================
def kfold(mdl, x_train, y_train, x_test):

    kf = KFold(n_splits=nfolds)

    train = np.zeros((x_train.shape[0],))
    test = np.zeros((x_test.shape[0],))
    test_kf = np.empty((nfolds, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        mdl.fit(x_tr, y_tr)
        train[test_index] = mdl.predict(x_te)
        test_kf[i, :] = mdl.predict(x_test)

    test[:] = test_kf.mean(axis=0)
    return train.reshape(-1, 1), test.reshape(-1, 1)


# =============================================================================
# Define the parameters required by each model in the stack
# =============================================================================
# Random Forest parameters
rf_params = {'n_estimators': 500, 'warm_start': True, 'max_depth': 6,
             'min_samples_leaf': 2, 'max_features': 'sqrt', 'verbose': 0}

# Extra Trees Parameters
et_params = {'n_estimators': 500, 'max_depth': 8,
             'min_samples_leaf': 2, 'verbose': 0}

# AdaBoost parameters
ada_params = {'n_estimators': 500, 'learning_rate': 0.75}

# Gradient Boosting parameters
gb_params = {'n_estimators': 500, 'max_depth': 5,
             'min_samples_leaf': 2, 'verbose': 0}

# Support Vector Classifier parameters
svc_params = {'kernel': 'linear', 'C': 0.025}

# =============================================================================
# Create the models using the helper function 'model'
# =============================================================================
rf = model(mdl=RandomForestClassifier, params=rf_params)
et = model(mdl=ExtraTreesClassifier, params=et_params)
ada = model(mdl=AdaBoostClassifier, params=ada_params)
gb = model(mdl=GradientBoostingClassifier, params=gb_params)
svc = model(mdl=SVC, params=svc_params)

# =============================================================================
# Train each of the models and save outputs as new features.
# Needs to convert dataframe values into numpy arrays first.
# =============================================================================
x_train = X_train.values
x_test = X_test.values

et_train, et_test = kfold(et, x_train, y_train, x_test)
rf_train, rf_test = kfold(rf, x_train, y_train, x_test)
ada_train, ada_test = kfold(ada, x_train, y_train, x_test)
gb_train, gb_test = kfold(gb, x_train, y_train, x_test)
sv_train, sv_test = kfold(svc, x_train, y_train, x_test)

print("First level training is complete")

# =============================================================================
# Second level training uses the outputs from the classifiers above
# =============================================================================
x_train = np.concatenate((et_train, rf_train, ada_train,
                          gb_train, sv_train), axis=1)
x_test = np.concatenate((et_test, rf_test, ada_test,
                         gb_test, sv_test), axis=1)
# =============================================================================
# Second level training using XGBClassifier
# =============================================================================
xgb_model = XGBClassifier(learning_rate=0.02, n_estimators=2000,
                          max_depth=4, min_child_weight=2, gamma=0.9,
                          subsample=0.8, colsample_bytree=0.8,
                          objective='binary:logistic', scale_pos_weight=1)
xgb_model.fit(x_train, y_train)
print("Second level training is complete")
predictions = xgb_model.predict(x_test)

# =============================================================================
# Save predictions in Kaggle format to be submitted to the competition
# =============================================================================
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your predictions were successfully saved!")
