# create a spammer identification using guassian mixture model (SI-GMM)

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score


# read the csv file and convert it to a numpy array
df = pd.read_csv('./spam-DATASET.csv', encoding='latin-1')
print(df.head())

# replace ham with 0 and spam with 1
df['v1'] = df['v1'].apply(lambda x: 0 if x == 'ham' else 1)

# drop the unnecessary columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# rename the columns
df = df.rename(columns={'v1': 'label', 'v2': 'text'})

print(df.head())

# convert the dataframe to a numpy array
data = df.to_numpy()

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data[:, 1], data[:, 0], test_size=0.2, random_state=42)

# create a count vectorizer
vectorizer = CountVectorizer()

# fit the vectorizer on the training data
vectorizer.fit(X_train)

# transform the training and testing data
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# create a Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

# fit the model on the training data
gmm.fit(X_train.toarray())

# predict the labels on the training data
y_pred = gmm.predict(X_train.toarray())

# calculate the accuracy score
accuracy = accuracy_score(y_train, y_pred)
print('Train Accuracy: ', accuracy)

# calculate the precision score
precision = precision_score(y_train, y_pred)
print('Precision: ', precision)

# calculate the recall score
recall = recall_score(y_train, y_pred)
print('Recall: ', recall)

# calculate the confusion matrix
confusion_matrix = confusion_matrix(y_train, y_pred)

# print the confusion matrix
print('Confusion Matrix: ')
print(confusion_matrix)

# predict the labels on the testing data
y_pred = gmm.predict(X_test.toarray())

# calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy: ', accuracy)

# calculate the precision score
precision = precision_score(y_test, y_pred)
print('Precision: ', precision)

# calculate the recall score
recall = recall_score(y_test, y_pred)
print('Recall: ', recall)

# calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ')
print(confusion_matrix)