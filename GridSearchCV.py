from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from IPython.display import display

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

pipe = Pipeline([('scale', StandardScaler()), ('model', KNeighborsRegressor())])

param_grid = {'model__n_neighbors': [1, 2, 3, 4, 5]}

mod = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, error_score='raise')

mod.fit(X, y)


dataframe = pd.DataFrame(mod.cv_results_)
#displayed in jupyter notebook
display(dataframe)
