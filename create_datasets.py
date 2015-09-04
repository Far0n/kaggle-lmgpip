import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

# LabelEncoder
train = pd.read_csv('../input/train.csv', index_col=None)
test = pd.read_csv('../input/test.csv', index_col=None)

train_cols = train.columns
test_cols = test.columns
labels = np.array(train.Hazard).ravel()
train_ids = np.array(train.Id).ravel()
test_ids = np.array(test.Id).ravel()

train.drop('Id', axis=1, inplace=True)
train.drop('Hazard', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:, i]) + list(test[:, i]))
    train[:, i] = lbl.transform(train[:, i])
    test[:, i] = lbl.transform(test[:, i])

train = np.column_stack((train_ids, labels, train))
test = np.column_stack((test_ids, test))

train = pd.DataFrame(train, columns=train_cols)
test = pd.DataFrame(test, columns=test_cols)

train['Id'] = train['Id'].astype(int)
train['Hazard'] = train['Hazard'].astype(int)
test['Id'] = test['Id'].astype(int)

train.to_csv('../input/train2.csv', index=None)
test.to_csv('../input/test2.csv', index=None)


# DictVectorizer
train = pd.read_csv('../input/train.csv', index_col=None)
test = pd.read_csv('../input/test.csv', index_col=None)

train_cols = train.columns
test_cols = test.columns
labels = np.array(train.Hazard).ravel().astype(int)
train_ids = np.array(train.Id).ravel().astype(int)
test_ids = np.array(test.Id).ravel().astype(int)

train.drop('Id', axis=1, inplace=True)
train.drop('Hazard', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train = train.T.reset_index(drop=True).to_dict().values()
test = test.T.reset_index(drop=True).to_dict().values()

vec = DictVectorizer(sparse=False)
train = vec.fit_transform(train)
test = vec.transform(test)

train = np.column_stack((train_ids, labels, train))
test = np.column_stack((test_ids, test))

train = pd.DataFrame(train, columns=['Id', 'Hazard'] + vec.get_feature_names())
test = pd.DataFrame(test, columns=['Id'] + vec.get_feature_names())

train['Id'] = train['Id'].astype(int)
train['Hazard'] = train['Hazard'].astype(int)
test['Id'] = test['Id'].astype(int)

train.to_csv('../input/train3.csv', index=None)
test.to_csv('../input/test3.csv', index=None)


# Factors to hazard mean
train = pd.read_csv('../input/train.csv', index_col=None)
test = pd.read_csv('../input/test.csv', index_col=None)

train_cols = train.columns
test_cols = test.columns
labels = train.Hazard.astype(int)
train_ids = train.Id.astype(int)
test_ids = test.Id.astype(int)

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

for feat in train.select_dtypes(include=['object']).columns:
    m = train.groupby([feat])['Hazard'].mean()
    train[feat].replace(m, inplace=True)
    test[feat].replace(m, inplace=True)

train = pd.concat((train_ids, train), axis=1)
test = pd.concat((test_ids, test), axis=1)

train.to_csv('../input/train4.csv', index=None)
test.to_csv('../input/test4.csv', index=None)
