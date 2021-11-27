import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


# creating the model
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=[1])
])

# compiling the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# creating some data
x = []
y = []
xs = np.array([])
ys = np.array([])

for i in range(1, 100001):
    x.append(i)
    y.append(i**2)

xs = np.append(xs, [x])
ys = np.append(ys, [y])

# normalising data
mean_xsdata = np.mean(xs)
mean_ysdata = np.mean(ys)

stdxs = np.std(xs)
stdys = np.std(ys)

for k in range(len(xs)):
    xs[k] = (xs[k] - mean_xsdata) / stdxs
    ys[k] = (ys[k] - mean_ysdata) / stdys




# training the model
model.fit(xs, ys, epochs=10)

# predicting the values
prediction = model.predict([2])
print(prediction)
321kabuniv123