import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mnistDataLoader import X_train,X_test,y_train,y_test
import numpy as np
import matplotlib.pyplot as plt

#Build model
tf.random.set_seed(1234)
backPropModel = Sequential(
    [
        tf.keras.Input(shape=(784,)),
        Dense(25,activation='relu'),
        Dense(15,activation='relu'),
        Dense(10,activation='linear'),
    ], name='hand_written_digit_classifier_model'
)
backPropModel.summary()


#Reshape data to fit model

# print("Shapes before reshape:" + "\n" + f"X_train,y_train{'\n'}{X_train,y_train}")
# print(X_train.shape,y_train.shape)
# print("Shapes before reshape:" + "\n" + f"X_test,y_test{'\n'}{X_test,y_test}")
# print(X_test.shape,y_test.shape)
# X_train = X_train.reshape(X_train.shape[0],-1)
# X_test = X_test.reshape(X_test.shape[0],-1)
# print("Shapes after reshape:" + "\n" + f"X_train,y_train{'\n'}{X_train,y_train}")
# print(X_train.shape,y_train.shape)
# print("Shapes after reshape:" + "\n" + f"X_train,y_train{'\n'}{X_test,y_test}")
# print(X_test.shape,y_test.shape)


#Train model
backPropModel.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
)
history = backPropModel.fit(X_train,y_train,epochs=50)

# Plot training/validation loss
# plt.figure(figsize=(8, 5))
# plt.plot(history.history["loss"], label="train_loss")

# if "val_loss" in history.history:
#     plt.plot(history.history["val_loss"], label="val_loss")

# plt.title("Model Loss During Training")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()


#Save model for export
backPropModel.save('backPropModel.keras')   #The model mostly struggles on 7 and 9, sometimes 4 
                                            #instances of live drawing



