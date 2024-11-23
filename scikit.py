import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()


X = diabetes.data
y = diabetes.target

# print("X: \n", X[:5])
# print("y: \n", y[:5])
#
# print("Feature: \n", diabetes.feature_names)
# print("Dataset Description: \n", diabetes.DESCR)


from sklearn.neighbors import KNeighborsRegressor

mod = KNeighborsRegressor() #model with no learning
#mod.predict(X) #has not been fitted yet so will result in error

mod.fit(X,y)

pred =  mod.predict(X) #now it works

plt.scatter(pred, y)
plt.show()
