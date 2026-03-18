import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.datasets import mnist

#Load the data: it is automatically split into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(f"{X_train.shape},{X_test.shape}")

#Preprocess the data: normalize and reshape
X_train = np.array(X_train).reshape(X_train.shape[0],-1)
X_test = np.array(X_test).reshape(X_test.shape[0],-1)
X_train = X_train / 255.0 # Each sample from the train and test set is a flatten out 28x28 grid of gray scale pixels
X_test = X_test / 255.0 



