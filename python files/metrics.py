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

from sklearn.linear_model import LogisticRegression

mod = LogisticRegression(class_weight={0:1,1:2}, max_iter=1000)

mod.predict(X,y)
