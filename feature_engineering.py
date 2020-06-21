# -*- coding: utf-8 -*-
"""
@author: surya

feature_engineering.py: developed for the Kaggle titanic challenge

input: dataframe (can be train/validation data or test data)
output: X - dataframe containing features after feature engineering
        y - list containing 'Survived' labels for train/validation data
          - set to 0 for test data  
          
Module contains two functions:  
    
plot_features(X) - to visualise relationship between different features and
the target labels in bar plot form. Defined as a separate function to enable
use at various stages of the workflow and support decision-making in feature
engineering

feature_eng(data) - the main feature engineering module. Developed specifically
for the Kaggle titanic challenge after close inspection of the training data
so as to perform as best as possible on the test data.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def plot_features(X):
    """
    plot as bar graphs, each feature against the survival probability. Can
    be used to visualise their relationship and consequently, the importance
    in training.
    """
    features = X.columns
    for feature in features:
        g = sns.catplot(x=feature, y="Survived", data=X, kind="bar",
                        height=6, palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")


def feature_eng(data, datatype):
    """
    Parameters
    ----------
    data : TYPE: Pandas dataframe
           DESCRIPTION: Contains training/validation data or test data

    Returns
    -------
    en_X : TYPE: Pandas dataframe
           DESCRIPTION: Contains features after feature engineering
    y : TYPE: Series object containing target labels
        DESCRIPTION: Contains labels if en_X is training or validation data;
                     else set to a dummy value of 0 for test data

    Missing entries are dealt with after closer inspection using the 
    plot_features function. 'Cabin' has a lot of missing entries but contains
    potentially useful information so the missing entries have been coded as
    'X' and used in training and predictions. Other missing entries have been 
    imputed (with the most frequent entry if categorical feature and the 
    median if numerical feature). Titles have been stripped out of names and
    used as a feature, after grouping all the rare ones into a single 
    category as well as grouping all females (Ms, Mrs etc.) together. Siblings
    and parent information have been combined to create a feature called 
    'Family Size'. All categorical features have been encoded into numbers (0, 
    1, 2, etc.) 
    """
# =============================================================================
#   Separate labels and features in train data. If test data, do nothing
# =============================================================================
    if 'Survived' in data.columns:
        data.dropna(axis=0, subset=['Survived'], inplace=True)
        y = data.Survived
    else:
        y = 0

# =============================================================================
#   Fill missing values with NaN
# =============================================================================
    data = data.fillna(np.nan)

# =============================================================================
#   Visualise missing values and personalise impute approach according to
#   feature
# =============================================================================
    print("Missing values counted in the {0} data: ".format(datatype))
    print(data.isnull().sum())

# =============================================================================
#   Replace cabin entry with the first letter of the cabin, if NaN, enter X
#   rather than impute
# =============================================================================
    data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in
                               data['Cabin']])

# =============================================================================
#   Impute object columns with most frequent value and numeric columns with
#   median
# =============================================================================
    mf_imputer = SimpleImputer(strategy='most_frequent')
    med_imputer = SimpleImputer(strategy='median')
    missing_cols = [col for col in data.columns if
                    data[col].isnull().sum() > 0]
    obj_col = [col for col in missing_cols if data[col].dtype == 'object']
    num_col = [col for col in missing_cols if data[col].dtype == 'int64' or
               data[col].dtype == 'float64']

    obj_imputed_data = data.copy()
    if obj_col:
        obj_imputed_data[obj_col] = pd.DataFrame(mf_imputer.fit_transform(
            data[obj_col]))
        obj_imputed_data.columns = data.columns
    num_imputed_data = obj_imputed_data.copy()
    if num_col:
        num_imputed_data[num_col] = pd.DataFrame(med_imputer.fit_transform(
            obj_imputed_data[num_col]))
        num_imputed_data.columns = data.columns

# =============================================================================
#   Plot a correlation matrix between different features to visualise the
#   correlation between them and 'Survived'
#   Uncomment the code below to plot - may make the code run slower
# =============================================================================
    # sns.heatmap(num_imputed_data.iloc[:, 1:].corr(),
    #            annot=True, fmt='.2f')
    # plot_features(num_imputed_data)

# =============================================================================
#   Feature engineer variables such as name by extracting titles
#   out of names etc.
# =============================================================================
    X = num_imputed_data.copy()

    # Name titles
    X["Title"] = X["Name"].str.extract('([A-Za-z]+\.)', expand=False)
    X.drop(['Name'], axis=1, inplace=True)
# =============================================================================
#   Visualise all present titles - only 4 are common - keep them and replace
#   others with 'rare' also group all title variations in spelling as well as
#   titles for same gender
#   Uncomment the respective lines to plot - may make the code run slower
# =============================================================================
    # g = sns.countplot(x="Title", data=X)
    # plt.setp(g.get_xticklabels(), rotation=45)
    titles_kept = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
    X.loc[~X["Title"].isin(titles_kept), "Title"] = "Rare"
    X['Title'] = X['Title'].map({'Mr.': 0, 'Mrs.': 1, 'Miss.': 1,
                                 'Master.': 2, 'Rare': 3})
    # plot_features(X)

# =============================================================================
#   SibSp and Parch - generate a 'Family_size' feature based on these
#   Uncomment the respective line to plot family size - may make the code
#   run slower
# =============================================================================
    X['fsz'] = X['SibSp'] + X['Parch']
    X.loc[X['fsz'] == 0, 'Family_size'] = 1
    X.loc[X['fsz'] == 1, 'Family_size'] = 2
    X.loc[X['fsz'] == 2, 'Family_size'] = 3
    X.loc[X['fsz'] == 3, 'Family_size'] = 3
    X.loc[X['fsz'] >= 4, 'Family_size'] = 4
    X['Family_size'].astype(int)
    X.drop(['fsz'], axis=1, inplace=True)
    # sns.catplot(x="Family_size", y="Survived", data=X, kind='bar')

# =============================================================================
#   Dividing fare and age into categories performed much worse than leaving
#   them as they are.
# =============================================================================

# =============================================================================
#   Label encode categorical columns
# =============================================================================
    label_encoder = LabelEncoder()
    en_X = X.copy()
    en_X['Sex'] = label_encoder.fit_transform(X['Sex'])
    en_X['Cabin'] = label_encoder.fit_transform(X['Cabin'])
    en_X['Embarked'] = label_encoder.fit_transform(X['Embarked'])

# =============================================================================
#   Drop columns containing no training information
# =============================================================================
    if 'Survived' in data.columns:
        en_X.drop(["PassengerId", "Ticket", "Survived"], axis=1,
                  inplace=True)
    else:
        en_X.drop(["PassengerId", "Ticket"], axis=1, inplace=True)

    return en_X, y
