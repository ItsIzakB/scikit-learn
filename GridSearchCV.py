from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target


pipe = Pipeline([('scale', StandardScaler), ('model', KNeighborsRegressor)])


