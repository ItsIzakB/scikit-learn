import numpy as np



arr = np.array(['low', 'low', 'high', 'medium']).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(arr)
print(encoded_data)


new_data = np.array(['zero']).reshape(1, -1)
encoded_new_data = encoder.transform(new_data) #will not work unless handle_unknown = 'ignore'

print('new data: ', encoded_new_data)
