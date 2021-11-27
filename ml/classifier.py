# building a classifier in Python
import sklearn
from sklearn import preprocessing
# import sklearn dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# print(label_names, "\n\n", labels, "\n\n", feature_names, "\n\n", features)

# organizing data into sets
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.40, random_state=42)

# building the model using Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

# evaluating the model and its accuracy
preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, preds))

# Naive Bayes Classifier is a classification technique used to build classifier using the Bayes Theorem where the
# assumption is that the predictors are independent i.e it assumes that the presence of a particular feature in a
# unrelated to the presences of any other feature.
# Types of Naive Bayes models under sklearn include Gaussian, Multinomial and Bernoulli
# Global AI Hub Main

