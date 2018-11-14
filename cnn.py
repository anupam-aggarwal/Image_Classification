#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import os

#initialising the CNN
classifier = Sequential()

# Convolution Layer
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation='relu'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten layer
classifier.add(Flatten())

#Fully Connected Layer
classifier.add(Dense(units = 1024,activation = 'relu'))
#classifier.add(Dense(units = 1,activation = 'sigmoid'))
classifier.add(Dense(units = 600,activation = 'softmax'))
#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )			#need to think about loss function

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(									#need to think about batch_size, class mode
        'image',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'image',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
print(training_set.class_indices)

classifier.fit_generator(															#need to tweak its params
        training_set,
        steps_per_epoch=512,
        epochs=25,
        validation_data=test_set,
        validation_steps=100)

#to save model to disk
# serialize model to JSON
model_json = classifier.to_json()
with open("../Model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("../Model/model.h5")
print("Saved model to disk")


'''
#to load model form json file
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''


