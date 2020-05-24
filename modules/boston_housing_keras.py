# import general libraries
import numpy as np
import matplotlib.pyplot as plt

# in case of running the code in Jupiter,
#    uncomment the line below
# %matplotlib inline

# import essential modules of keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import MaxPooling2D, Convolution2D
from keras.utils import np_utils
from keras.optimizers import SGD

# load data from the basic datasets of keras
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# build model
model = Sequential()
model.add(Dense(14, input_dim=13, kernel_initializer="normal", activation="relu"))
model.add(Dense(8, activation="relu", use_bias=True))
model.add(Dense(8, activation="relu", use_bias=True))
model.add(Dense(1))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error", optimizer="adam")

# summary of the total neural network
print(model.summary())

# train model using given datasets
model.fit(X_train, y_train, epochs=200)

# predict and compare to the actual values
predicted = model.predict(X_test)
print("Actual (first 5): {}".format(y_test[:5]))
