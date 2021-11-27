#support vector machine-supervised machine learning algorithm that can be used for both regression and classification.
#Plots each data item as a point in n-dimensional space with the value of each feature being the value of a particular
#coordinate
#Building an SVM classifier
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
#loading the input data
iris = datasets.load_iris()
#taking first 2 features
X = iris.data[:, :2]
y = iris.target
#plotting the svm boundaries with original data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
#giiving the value of regularization parameter
C = 1.0
#creating the svm clssifier object
Svc_classifier = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
Z = Svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.savefig("svm-pic.png")
plt.show()
