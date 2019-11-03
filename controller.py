# -*- coding: utf-8 -*-

from flask import Flask, request
from keras.models import model_from_json      
from keras.preprocessing import image
#from keras.applications import imagenet_utils
#import tensorflow as tf
import numpy as np

controller = Flask(__name__)           
classifier = None

    
@controller.route("/predict", methods=['POST'])                  
def predict():
    print("--- REQUEST RECEIVED ---")
    img = request.files['photo']
    img = prepare_image(img)
    result = classifier.predict(img)
    result = get_class_name(result)
    print(result)
    return result

def prepare_image(img):
    img = image.load_img(img, target_size = (128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    
    #test_image = imagenet_utils.preprocess_input(test_image)
    return img

def load_model():
    global classifier
    json_file = open('cnn_json2', 'r')
    classifier_json = json_file.read()
    json_file.close()
    classifier = model_from_json(classifier_json)
    classifier.load_weights('classifier2.h5')
    #classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("Classifier Ready")
    return classifier

def get_class_name(result):
    print(result)
    if result[0][0] == 1:
        return "Airplane"
    elif result[0][1] == 1:
        return "Camera"
    elif result[0][2] == 1:
        return "Chair"
    elif result[0][3] == 1:
        return "Cougar"
    elif result[0][4] == 1:
        return "Crocodile"
    elif result[0][5] == 1:
        return "Dog"
    elif result[0][6] == 1:
        return "Laptop"
    elif result[0][7] == 1:
        return "Pizza"
    elif result[0][8] == 1:
        return "Scissors"
    elif result[0][9] == 1:
        return "Watch"

   

if __name__ == "__main__":
    print("Running app and model")
    load_model()
    controller.run(debug = False, threaded = False)                  
    
