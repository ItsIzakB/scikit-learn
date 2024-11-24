from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data
y=diabetes.target

#preprocessing and pipeline
pipe = Pipeline([
                ("scale", StandardScaler()),
                 ("model", KNeighborsRegressor())
                 ])

pipe.fit(X,y)

pred = pipe.predict(X)

plt.plot(pred,y, 'o')
plt.xlabel('pred')
plt.ylabel('actual')

plt.show()
