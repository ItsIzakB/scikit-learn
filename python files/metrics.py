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

mod.fit(X,y)
ans  = mod.predict(X).sum
print(ans)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},
    cv=4,
    n_jobs=-1
)

grid.fit(X,y)

print(grid.cv_results_)

pd.DataFrame(grid.cv_results_)


