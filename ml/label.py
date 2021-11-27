import numpy as np
from sklearn import preprocessing

#sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
#creating & training label encoder object
encoder = preprocessing.LabelEncoder()
print(encoder.fit(input_labels))
#encoding a set of labels/checking performance by encoding random ordered list
test_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoded_values = encoder.transform(test_labels)
print(test_labels)
#getting the word labels converted to numbers
print(list(encoded_values))
#checking performance by decoding random set of numbers
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print(decoded_list)