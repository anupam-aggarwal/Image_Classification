#file to load a model and predict the class

#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

#to load model form json file

# load json and create model
json_file = open('../Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../Model/model.h5")
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X = np.load("Xdata.npy")
Y = np.load("Ydata.npy")
Y = np.eye(np.max(Y)+1)[Y]

score = model.evaluate(X, Y, verbose=1)
for i in range(0,len(model.metrics_names)):	
	print("%s: %.2f%%" % (model.metrics_names[i], score[i]*100))
