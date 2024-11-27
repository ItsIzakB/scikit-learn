import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('../CSV_files/drawndata2.csv')

X = data[['x', 'y']].values
y = data['z'] == 'a'
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
