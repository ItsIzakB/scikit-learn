import kagglehub
import pandas as pd
#
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#
# print(path)

file = pd.read_csv('../CSV_files/creditcard.csv')

X = file.drop(columns=['Time', 'Amount', 'Class'])
y = file['Class'].values

from sklearn.linear_model import LinearRegression

mod = LinearRegression()

mod.fit(X, y)

pred = mod.predict(X)
numFraud = pred.sum()

print(numFraud)
