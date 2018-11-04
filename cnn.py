#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifier = Sequential()

# Convolution Layer
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation='relu'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten layer
classifier.add(Flatten())

#Fully Connected Layer
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 1,activation = 'softmax'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )			#need to think about loss function

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(									#need to think about batch_size, class mode
        'img',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'img',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(															#need to tweak its params
        training_set,
        steps_per_epoch=551,
        epochs=25,
        validation_data=test_set,
        validation_steps=100)
