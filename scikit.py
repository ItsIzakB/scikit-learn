from sklearn.datasets import load_diabetes

diabetes = load_diabetes()


X = diabetes.data
y = diabetes.target

print("X: \n", X[:5])
print("y: \n", y[:5])

print("Feature: \n", diabetes.feature_names)
print("Dataset Description: \n", diabetes.DESCR)

