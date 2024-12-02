import kagglehub
import pandas as pd
#
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#
# print(path)

file = pd.read_csv('../CSV_files/creditcard.csv')

print(file.head(3))
