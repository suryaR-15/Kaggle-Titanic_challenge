# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:45:57 2020

@author: surya
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

# read in and look at training data

train_data = pd.read_csv(train_data_path)
# print(train_data.columns)
# print(train_data.describe())
# print(train_data.head())
# print(train_data.shape)

train_data.dropna(axis = 0, subset = ['Survived'], inplace = True)
y = train_data.Survived
train_data.drop(['Survived'], axis = 1, inplace = True)

# use only numerical and categorical columns for training the model and making predictions

num_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in train_data.columns if train_data[col].nunique() < 10 and 
            train_data[col].dtype == 'object']
train_cols = num_cols + cat_cols

X = train_data[train_cols].copy()


# define pipelines for numerical and categorical data to transform into something more 
# easy for the model to learn

num_transformer = SimpleImputer(strategy = 'mean')
cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
    ])

# pre-processing that transforms the data depending on whether numerical or categorical

preprocessor = ColumnTransformer(transformers = [
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
    ])

# define the model

model = RandomForestClassifier(n_estimators = 200, random_state = 1)

# define a workflow for the data that includes pre-processing and model fitting

my_pipeline = Pipeline(steps = [
    ('preprocessing', preprocessor),
    ('model', model)
    ])

# train the model using the pipeline and cross-validation method
 
scores = cross_validate(my_pipeline, X, y, cv = 5, scoring = 'accuracy', return_estimator = True)
print(scores.keys())
print("Test scores:")
print(scores['test_score'])
max_acc = np.argmax(scores['test_score'])+1  # index of maximum accuracy estimator
print("Maximum accuracy attained by estimator number " + format(max_acc))

best_model = scores['estimator'][max_acc - 1]

# run the best model on test_data

test_data = pd.read_csv(test_data_path)
X_test = test_data[train_cols].copy()
predictions = best_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index = False)
print("Your submission was successfully saved!")

