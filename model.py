from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocesing(new_image):
    img = image.load_img(new_image, target_size=(150, 150))
    array_img = image.img_to_array(img)
    dimen = np.expand_dims(array_img, axis=0)
    dimen /= 255.0
    return dimen

def prediction(model, input_data):
    preprocessed_image = preprocesing(input_data)
    predict = model.predict(preprocessed_image)
    
    if predict < 0.5:
        print("The image is predicted to be a Cat.")
    else:
        print("The image is predicted to be a Dog.")
    
    return predict

new_image = r"C:\Users\rajes\Datascience_jp\Day20\Animals\test_data\cats\0_0016.jpg"  
model = load_model('my_model.h5')  

prediction(model, new_image)
