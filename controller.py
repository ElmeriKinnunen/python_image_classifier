# -*- coding: utf-8 -*-

from flask import Flask   
from keras.models import model_from_json      
from keras.preprocessing import image
#from keras.applications import imagenet_utils
#import tensorflow as tf
import numpy as np

controller = Flask(__name__)           
classifier = None

    
@controller.route("/predict")                  
def predict():              
    test_image = prepare_image()   
    result = classifier.predict(test_image)
    result = get_class_name(result)
    print(result)
    return "Result = " + result

def prepare_image():
     #pizza = 'dataset/newpics/pizza/images401.jpg'
    scissors = 'dataset/test_set/camera/960x0.jpg'     
    test_image = image.load_img(scissors, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #test_image = imagenet_utils.preprocess_input(test_image)
    return test_image

def load_model():
    global classifier
    json_file = open('cnn_json', 'r')
    classifier_json = json_file.read()
    json_file.close()
    classifier = model_from_json(classifier_json)
    classifier.load_weights('classifier.h5')
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
    
