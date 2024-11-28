import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('../CSV_files/drawndata2.csv')

X = data[['x', 'y']].values
y = data['z'] == 'a'
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

#
# pipe = Pipeline([
#     ('scale', QuantileTransformer(n_quantiles= 1000)),
#     ('model', LogisticRegression())
# ])

#let's try polynomialfeatures

from sklearn.preprocessing import PolynomialFeatures

pipe = Pipeline([
    ('scale', PolynomialFeatures()),
    ('model', LogisticRegression())
])

pipe.fit(X,y)
pred = pipe.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.show()
