# Image_Classification

This project contains the code for a CNN Model which is used to classify the sanskrit letters which belong to 600 different classes.
This model is to used in a bigger project of SanskritOCR for leteer identification

The data is fed in the form of numpy array files. 
The results of the model are around 82% accuracy on the Validation Set.

cnn.py - Python file to run the model
predictions.py - Python file to make predictions

./Results/Final_model.h5, ./Results/Final_model.json - files for model architeture and weights.

# Dependecies
The code is mainly written in Keras, a high level api for Tensorflow.
So requirements will be
  1. Python 3.5
  2. Numpy,Matplotlib
  3. Tensorflow
  4. Keras

