import kagglehub
import pandas as pd
import numpy as np
#
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#
# print(path)

file = pd.read_csv('../CSV_files/creditcard.csv')

X = file.drop(columns=['Time', 'Amount', 'Class'])
y = file['Class'].values

print(f'shape of x: {X.shape} and shape of y: {y.shape}, '
      f'fraud = {y.sum()}')

from sklearn.linear_model import LinearRegression

mod = LinearRegression(class_weight)

mod.fit(X, y)

pred = mod.predict(X)
numFraud = pred.sum()

print(numFraud)
