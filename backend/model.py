import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mnistDataLoader import X_train,y_train

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

#Train model
backPropModel.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
)
backPropModel.fit(X_train,y_train,epochs=50)

#Save model for export
backPropModel.save('backPropModel.keras')   #The model mostly struggles on 7 and 9, sometimes 4 
                                            #instances of live drawing



