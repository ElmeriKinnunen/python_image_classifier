# -*- coding: utf-8 -*-

from keras.models import model_from_json

json_file = open('cnn_json2', 'r')
classifier_json = json_file.read()
json_file.close()

classifier = model_from_json(classifier_json)

classifier.load_weights('classifier.h5')

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

pizza = 'dataset/newpics/pizza/images401.jpg'
scissors = 'dataset/newpics/scissors/images399.jpg'


#Predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img(pizza, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
