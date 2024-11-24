from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load diabetes dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# Create the pipeline with instantiated objects for both steps
pipe = Pipeline([('scale', StandardScaler()), ('model', KNeighborsRegressor())])

# Perform grid search with the correct parameter grid
param_grid = {'model__n_neighbors': [1, 2, 3, 4, 5]}

mod = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, error_score='raise')

# Fit the model
mod.fit(X, y)

# Output the results
print(mod.cv_results_)
