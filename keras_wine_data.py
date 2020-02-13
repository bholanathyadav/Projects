# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:16:06 2020

@author: Bhola Nath Yadav
"""

from numpy import loadtxt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
np.random.seed(3)

# number of wine classes
classification = 3

# load dataset
dataset = np.loadtxt('C:\\Users\\CA DHEERAJ\\wine.csv', delimiter=",")

# split dataset into sets for testing and training
X = dataset[:,1:14]
Y = dataset[:,0:1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

# convert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, classification)
y_test = keras.utils.to_categorical(y_test-1, classification)

# creating model
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(classification, activation='softmax'))

# compile and fit model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=25, epochs=1000, validation_data=(x_test, y_test))