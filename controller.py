# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from keras.models import model_from_json      
from keras.preprocessing import image

import numpy as np

controller = Flask(__name__)           
classifier = None

@controller.route("/predict", methods=['POST'])                  
def predict():
    print("--- REQUEST RECEIVED ---")
    img = request.files['photo']
    img = prepare_image(img)
    result = classifier.predict(img)
    return get_class_name(result)
    

def prepare_image(sent_img):
    img = image.load_img(sent_img, target_size = (128, 128))
    img = image.img_to_array(img)/255
    img = np.expand_dims(img, axis = 0)
    return img

def load_model():
    global classifier
    json_file = open('cnn_json', 'r')
    classifier_json = json_file.read()
    json_file.close()
    classifier = model_from_json(classifier_json)
    classifier.load_weights('classifier.h5')
    print("Classifier Ready")
    return classifier

def get_class_name(result):
    result_to_class = ['Cat', 'Cup', 'Dog', 'Laptop', 'Pizza', 'Plant', 'Scissors', 'Watch']
    index = np.argsort(result[0,:])
   
    #print(result_to_class[index[7]], ' Probability', result[0, index[7]] )
    #print(result_to_class[index[6]], ' Probability', result[0, index[6]] )
    #print(result_to_class[index[5]], ' Probability', result[0, index[5]] )
    #print(result_to_class[index[4]], ' Probability', result[0, index[4]] )
    
    predResult1 = '{0:.3g}'.format(result[0, index[7]] * 100)
    predResult2 = '{0:.3g}'.format(result[0, index[6]] * 100)
    predResult3 = '{0:.3g}'.format(result[0, index[5]] * 100)
    predResult4 = '{0:.3g}'.format(result[0, index[4]] * 100)
    
    response = jsonify({
            "pred1": { result_to_class[index[7]]: str(predResult1)},
            "pred2": { result_to_class[index[6]]: str(predResult2)},
            "pred3": {result_to_class[index[5]]: str(predResult3)},
            "pred4": {result_to_class[index[4]]: str(predResult4)}
            })
    return response

if __name__ == "__main__":
    print("Running")
    load_model()
    controller.run(debug = False, threaded = False, host='0.0.0.0')                  
    
