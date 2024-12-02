import kagglehub
import pandas as pd
#
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#
# print(path)

file = pd.read_csv('../CSV_files/creditcard.csv')

X = file.drop(columns=['Time', 'Amount', 'Class'])
y = file['Class'].values

