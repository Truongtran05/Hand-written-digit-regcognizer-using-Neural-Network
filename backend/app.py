from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64, io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from tensorflow import keras

app = Flask(__name__)
CORS(app)

model = keras.models.load_model(
    os.path.join(BASE_DIR,"..","backPropModel.keras")
)

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    data = request.json['image']               # base64 string from canvas
    img_bytes = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = img.resize((280,280))
    img_array = np.array(img)
    # print(img_array)
    
    #Centering input digit 
    #Crop image
    img_pixels = img_array > 50
    rows_have_ink = np.any(img_pixels,axis=1)
    cols_have_ink = np.any(img_pixels,axis=0)

    if not np.any(rows_have_ink) or not np.any(cols_have_ink): #Case user drawn nothing
        return jsonify({'digit': '?', 'confidence': 0, 'mnist_image': None})

    x_min,x_max = np.where(cols_have_ink)[0][[0,-1]]
    y_min,y_max = np.where(rows_have_ink)[0][[0,-1]]
    img_crop = img_array[y_min:y_max+1,x_min:x_max+1]  

    #Resize to a 20x20 digit image
    img_center = Image.fromarray(img_crop)
    img_center = img_center.resize((20,20))
    img_center_array = np.array(img_center)

    #Put the 20x20 image in center of 28x28 blank background
    centered_image_array = np.zeros((28,28))
    x_offset = 4
    y_offset = 4 
    centered_image_array[x_offset:x_offset+20, y_offset:y_offset+20] = img_center_array

    #Display input image for debug
    # centered_image = Image.fromarray(centered_image_array)
    # plt.figure(figsize=(10,3)) 
    # plt.imshow(centered_image,cmap='gray')
    # plt.title(f"Input from client")
    # plt.show()

    # Build a displayable 28x28 image preview of the exact model input.
    preview_array = centered_image_array.astype(np.uint8)
    preview_img = Image.fromarray(preview_array, mode='L')
    preview_buffer = io.BytesIO()
    preview_img.save(preview_buffer, format='PNG')
    preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode('utf-8')
    preview_data_url = f"data:image/png;base64,{preview_base64}"

    #Reshape and normalize model input 
    centered_image_array = centered_image_array.reshape(1,784)/255.0 
    prediction = model.predict(centered_image_array, verbose=0)
    probs = keras.activations.softmax(prediction).numpy()
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100

    return jsonify({
        'digit': digit,
        'confidence': round(confidence, 2),
        'mnist_image': preview_data_url
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
