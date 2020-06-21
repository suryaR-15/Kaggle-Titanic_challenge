# -*- coding: utf-8 -*-
"""
@author: surya

Kaggle titanic challenge: https://www.kaggle.com/c/titanic/overview

XGB Classifier has been used after feature engineering
Cross-validation approach used; best model used to predict labels on test data

Kaggle accuracy score achieved using this version of the code - 0.78468
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from feature_engineering import feature_eng

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

# =============================================================================
# Read in and look at training data - uncomment as required
# =============================================================================
train_data = pd.read_csv(train_data_path)
# print(train_data.columns)
# print(train_data.describe())
# print(train_data.head())
# print(train_data.shape)

# =============================================================================
# Feature engineering defined as a separate function - specific to this
# dataset
# =============================================================================
X, y = feature_eng(train_data, 'training')

# =============================================================================
# Define the model
# =============================================================================
model = XGBClassifier(n_estimators=1000, learning_rate=0.01,
                      random_state=1, max_depth=5, gamma=0.6,
                      subsample=1, min_child_weight=5, colsample_bytree=0.9,
                      reg_alpha=0, reg_lambda=0.5)

# =============================================================================
# Define a workflow for the data that includes pre-processing and
# model fitting
# =============================================================================
my_pipeline = Pipeline(steps=[
    ('model', model)
    ])

# =============================================================================
# Train the model using the pipeline and cross-validation method
# Use indexing to access the estimator with the best test score and use
# it to predict labels for test data
# =============================================================================
scores = cross_validate(my_pipeline, X, y, cv=5, scoring='f1',
                        return_estimator=True)
print("F1 scores on validation data:")
print(scores['test_score'])
max_acc = np.argmax(scores['test_score']) + 1
print("Maximum F1 score attained by estimator number " + format(max_acc))

best_model = scores['estimator'][max_acc - 1]

# =============================================================================
# Run the best model on Kaggle test_data
# =============================================================================
test_data = pd.read_csv(test_data_path)
test_data_copy = test_data.copy()
print("\nMaking predictions on test data using estimator\
      {0}...".format(max_acc))
X_test, y_dummy = feature_eng(test_data, 'test')
predictions = best_model.predict(X_test)

# =============================================================================
# Save predictions in Kaggle format to be submitted to the competition
# =============================================================================
output = pd.DataFrame({'PassengerId': test_data_copy.PassengerId,
                       'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your predictions were successfully saved!")
