from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('my_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    array_img = image.img_to_array(img)
    dimen = np.expand_dims(array_img, axis=0)
    dimen /= 255.0
    return dimen

def predict_image(img_path):
    preprocessed_image = preprocess_image(img_path)
    predict = model.predict(preprocessed_image)
    return "Cat" if predict < 0.5 else "Dog"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        result = predict_image(filepath)
        return render_template('result.html', prediction=result, image_path='uploads/' + file.filename)

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
