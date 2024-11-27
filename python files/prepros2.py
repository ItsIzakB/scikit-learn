import pandas as pd
import numpy as np

import matplotlib.pylab as plt


data = pd.read_csv('../CSV files/drawndata1.csv')

# print(data.head(3))


X = data[['x', 'y']].values
y = data['z'] == 'a'

# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()

#scaling with StandardScalar -- issue is with outliers

# from sklearn.preprocessing import StandardScaler
#
# X_new = StandardScaler().fit_transform(X)
#
# plt.scatter(X_new[:,0], X_new[:,1], c=y)
# plt.show()

#scaling with QuantileTransformer

from sklearn.preprocessing import QuantileTransformer

X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)

plt.scatter(X_new[:,0], X_new[:,1], c=y)
plt.show()
