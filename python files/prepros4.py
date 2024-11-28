import numpy as np



arr = np.array(['low', 'low', 'high', 'medium']).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore')
encoded_data = encoder.fit_transform(arr)
print(encoded_data)
