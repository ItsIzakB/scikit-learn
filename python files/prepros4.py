import numpy as np



arr = np.array(['low', 'low', 'high', 'medium']).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown= 'ignore')
encoder.fit_transform(arr)

print(arr)
