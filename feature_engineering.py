# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 06:19:40 2020

@author: surya
"""

import pandas as pd
import numpy as np
from operator import attrgetter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def feature_eng(data, verbose = True):
    ### separate labels and features in train data. If test data, do nothing
    if 'Survived' in data.columns:
        data.dropna(axis = 0, subset = ['Survived'], inplace = True)
        y = data.Survived
    else:
        y = 0
        
    ### drop irrelevant columns e.g. PassengerId, Ticket and convert columns to more meaningful entries
    ### e.g. names to titles of names; age and fare into bands.
    data.drop(['PassengerId'], axis = 1, inplace = True)
    data.drop(['Ticket'], axis = 1, inplace = True)
    data["Name_title"] = data["Name"].str.extract('([A-Za-z]+\.)',expand=False)
    data.drop(['Name'], axis = 1, inplace = True)
    data['Age_band'] = pd.qcut(data["Age"], 20)
    data.drop(['Age'], axis = 1, inplace = True)
    data["Fare_band"] = pd.qcut(data["Fare"], 20)
    data.drop(['Fare'], axis = 1, inplace = True)
    data['Embarked'] = data['Embarked'].astype(str)
    
    # convert age and fare bands into two columns containing min. and max. of each category - both not required for training
    #data['Age_left'] = pd.to_numeric(data['Age_band'].map(attrgetter('left')))
    data['Age_right'] = pd.to_numeric(data['Age_band'].map(attrgetter('right')))
   # data['Fare_left'] = pd.to_numeric(data['Fare_band'].map(attrgetter('left')))
    data['Fare_right'] = pd.to_numeric(data['Fare_band'].map(attrgetter('right')))
    data.drop(['Age_band', 'Fare_band'], axis = 1, inplace = True)
    
    ### group passengers by each feature and check if it has an effect on the survival rate
    ### use as needed for feature engineering
# =============================================================================
#     features = data.columns
#     print("Features available in training data: {0}".format(features))
#     for feature in features:
#         if feature != 'Survived':
#             print("Analysing feature {0}".format(feature))
#             if data[feature].isna().sum() >= data.shape[0]/2:
#                 print("{0} data dropped - too many missing entries".format(feature))
#                 data.drop([feature], axis = 1, inplace = True)
#             elif 'Survived' in data.columns:
#                 grouped = data.groupby(feature)['Survived'].mean()
#                 print(grouped)
#             else:
#                 pass 
# =============================================================================
    ### drop 'Survived' column from training data after visualising as groups
    if 'Survived' in data.columns:
        data.drop(['Survived'], axis = 1, inplace = True)
    
    ### check which features contain NaN values and drop if too many; else: encode and impute
    features = data.columns
    for feature in features:
        n = data[feature].isna().sum()
        if n >= data.shape[0]/2:
            if verbose == True:
                print("{0} contains {1} missing entries. Dropping this feature...".format(feature, n))
            data.drop([feature], axis = 1, inplace = True)
    encoded_data = data.copy()  
    features = encoded_data.columns  
    if verbose == True:
        print("Pre-processing remaining features and training...")
    for feature in features:
            n = data[feature].isna().sum()
            label_encoder = LabelEncoder()
            encoded_data[feature] = label_encoder.fit_transform(data[feature])
    imputer = SimpleImputer(strategy = 'most_frequent')
    imputed_data = encoded_data.copy()
    imputed_data = pd.DataFrame(imputer.fit_transform(encoded_data))
    imputed_data.columns = encoded_data.columns
    
    X = imputed_data.copy()
    return X, y
