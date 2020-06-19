# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:45:57 2020

@author: surya
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

### read in and look at training data

train_data = pd.read_csv(train_data_path)
# print(train_data.columns)
# print(train_data.describe())
# print(train_data.head())
# print(train_data.shape)

### feature engineering defined as a separate function - specific to this dataset

from feature_engineering import feature_eng
X, y = feature_eng(train_data)

### define the model

model = XGBClassifier(n_estimators = 1000, learning_rate = 0.01, random_state = 1,
                      max_depth = 20, gamma = 0.5, subsample = 0.5)

### define a workflow for the data that includes pre-processing and model fitting

my_pipeline = Pipeline(steps = [
    ('model', model)
    ])

### train the model using the pipeline and cross-validation method
  
scores = cross_validate(my_pipeline, X, y, cv = 5, scoring = 'f1', return_estimator = True)
print("F1 scores on validation data:")
print(scores['test_score'])
max_acc = np.argmax(scores['test_score'])+1  # index of maximum accuracy estimator
print("Maximum F1 score attained by estimator number " + format(max_acc))

best_model = scores['estimator'][max_acc - 1]

### run the best model on Kaggle test_data

test_data = pd.read_csv(test_data_path)
test_data_copy = test_data.copy()
print("\nMaking predictions on test data using estimator {0}...".format(max_acc))
X_test, y_dummy = feature_eng(test_data, verbose = False)
predictions = best_model.predict(X_test)

### save predictions in Kaggle format to be submitted to the competition

output = pd.DataFrame({'PassengerId': test_data_copy.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index = False)
print("Your predictions were successfully saved!")
