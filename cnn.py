#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifier = Sequential()

# Convolution Layer
classifier.add(Convolutio2D(32,3,3,input_shape = (64,3,3), activation='relu'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten layer
classifier.add(Flatten())

#Fully Connected Layer
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 602,activation = 'softmax'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = '', metrics = ['accuracy'] )

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'image/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'image/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=test_set,
        validation_steps=800)