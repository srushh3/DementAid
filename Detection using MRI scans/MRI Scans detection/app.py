from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='templates')

model = load_model('model.h5')
model.make_predict_function()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(120, 120), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_dementia_stage(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        brain_scan = request.files['brain_scan']
        brain_scan.save('uploads/' + brain_scan.filename)
        prediction = predict_dementia_stage('uploads/' + brain_scan.filename)
        # categories = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]
        # "Prediction: {0}".format(categories[np.argmax(prediction)])
        
        if prediction <=0.5:
            dementia_stage = 'Demented ---- Take care! Do make use of the app to ease your life'
        else:
            dementia_stage = 'Non-Demented ---- Congratulations! You are at the peak of your health'
        return render_template('results.html', dementia_stage=dementia_stage)

       

        
    return render_template('index.html')
@app.route('/dementia.jpg')
def serve_image():
    return send_from_directory('.','dementia.jpg')

if __name__ == '__main__':
    app.run(debug=True)
