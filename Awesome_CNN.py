# -*- coding: utf-8 -*-

# building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize the CNN
classifier = Sequential()

# 1st Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 3nd Convolution 
classifier.add(Conv2D(64, (6, 6), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 10, activation = "softmax"))

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



#Image preprocessing and altering
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128), 
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128), 
                                            batch_size = 32,
                                            class_mode = 'categorical')
#Fitting the images to the model
classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 700)

#Save model & weights
classifier_json = classifier.to_json()
with open('cnn_json', 'w') as json_file:
    json_file.write(classifier_json)

classifier.save_weights("classifier.h5")


#PREDICTIONS
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/test_set/crocodile/image_0046.jpg',  target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

#UNCOMMENT TO SEE CNN CATEGORY OUTPUT INDICES
training_set.class_indices























