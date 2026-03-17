import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
import matplotlib.pyplot as plt
from mnistDataLoader import X_test,y_test


model = keras.models.load_model('backPropModel.keras')

test_prediction = np.argmax(model.predict(X_test),axis=1)
precision = np.count_nonzero(test_prediction == y_test)/np.size(y_test)
print(f"Precision on test data: {precision*100.0}%")

# plt.figure(figsize=(10,5))
# for i in range(5):
#     plt.subplot(1,5,i+1)
#     plt.imshow(X_test[i].reshape(28,28),cmap='gray')
#     plt.title(f"lable:{test_prediction[i]}")
#     plt.axis('off') 
# plt.show()
