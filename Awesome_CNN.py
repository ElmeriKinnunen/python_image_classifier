# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation="relu"))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(Conv2D(64, (6, 6), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 256, activation = "relu"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 8, activation = "softmax"))

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


#Image preprocessing and altering
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                 target_size = (128, 128), 
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                            target_size = (128, 128), 
                                            batch_size = 32,
                                            class_mode = 'categorical')
classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 700)

#Save model & weights
classifier_json = classifier.to_json()
with open('cnn_demo', 'w') as json_file:
    json_file.write(classifier_json)

classifier.save_weights("classifier.h5")

#UNCOMMENT TO SEE CNN CATEGORY OUTPUT INDICES
#training_set.class_indices























