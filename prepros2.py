import pandas as pd
import numpy as np

import matplotlib.pylab as plt


data = pd.read_csv('drawndata1.csv')

print(data.head(3))


X = data[['x', 'y']].values
y = data['z'] == 'a'

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
