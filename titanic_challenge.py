# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:28:04 2020

@author: surya
"""

# importing required modules

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# define data paths

train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'

# read in and look at training data

train_data = pd.read_csv(train_data_path)
# print(train_data.columns)
# print(train_data.describe())
# print(train_data.head())
# print(train_data.shape)

# feature engineering to improve the performance of the model

train_data['Sex'].replace(['male', 'female'],[0, 1], inplace = True) # convert gender to 0's and 1's
train_data['Age'].replace(np.nan, 0, inplace = True) # convert age < 0 to 0
train_data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace = True) # convert ports to numerical
train_data['Embarked'].replace(np.nan, 0, inplace = True) # impute NaNs to 0
# convert people to children (0), adults (1) and the elderly (2)
train_data.loc[train_data.Age < 18, 'Age'] = 0 
train_data.loc[(train_data.Age >= 18) & (train_data.Age < 55), 'Age'] = 1
train_data.loc[train_data.Age >= 55, 'Age'] = 2  

# create features X and labels y from train_data; try experimenting with
# different combinations of features

all_features = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']  # not 'all' features, some are skipped e.g. Name
selected_features = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked']
y = train_data['Survived']
X = train_data[selected_features]
# print(y)
# print(X.columns)
# print(y.shape)
# print(X.shape)

# split train_data into training set and validation set

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 1)

# train the random forest classifier using train_X and train_y

model = RandomForestClassifier(n_estimators = 500, min_samples_split = 3,
                               random_state = 1)
model.fit(train_X, train_y)
train_predictions = model.predict(train_X)
print("Training accuracy is: ")
C1 = confusion_matrix(train_y, train_predictions)
train_accuracy = (C1[1][1] + C1[0][0])/train_y.shape[0]
print(train_accuracy)

# test the model using the validation set

val_predictions = model.predict(val_X)
print("Validation accuracy is: ")
C2 = confusion_matrix(val_y, val_predictions)
val_accuracy = (C2[1][1] + C2[0][0])/val_y.shape[0]
print(val_accuracy)

# see other performance metrics e.g. precision, recall, f1-score

print("Training summary metrics")
print(classification_report(train_y, train_predictions))
print("Validation summary metrics")
print(classification_report(val_y, val_predictions))


# use the model on test data

test_data = pd.read_csv(test_data_path)

# use same processing on test data and features
test_data['Sex'].replace(['male', 'female'],[0, 1], inplace = True) # convert gender to 0's and 1's
test_data['Age'].replace(np.nan, 0, inplace = True) # convert age < 0 to 0
test_data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace = True) # convert ports to numerical
test_data['Embarked'].replace(np.nan, 0, inplace = True) # impute NaNs to 0
# convert people to children (0), adults (1) and the elderly (2)
test_data.loc[train_data.Age < 18, 'Age'] = 0 
test_data.loc[(test_data.Age >= 18) & (test_data.Age < 55), 'Age'] = 1
test_data.loc[test_data.Age >= 55, 'Age'] = 2  

# extra processing step since test_data had a fare entry of NaN - imputed to average fare
test_data['Fare'].replace(np.nan, np.mean(test_data['Fare']), inplace = True)
test_X = test_data[selected_features]

test_predictions = model.predict(test_X)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
