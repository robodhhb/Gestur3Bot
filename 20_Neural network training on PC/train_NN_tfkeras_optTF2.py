# -*- coding: utf-8 -*-
# Train Gestur3Bot's NN with TensorFlow internal Keras optimized

#This program uses the internal Keras library of TensorFlow for the training of the neural network.
#The architecture of the NN is 32-20-7. It uses Adam-Optimizer for better results.
#Input and output data use the subdirectory "Data". During the training the program checks every epoch if the accuracy of the model has improved. If this is true the model (checkpoint) is saved and later on reloaded to save the weights and biases in a csv and rtf-file for Lego Mindstorms.

#This programm needs the following installation for saving the checkpoints: pip3 install h5py

#Version 1.1 for TensorFlow 2.X and TF Keras

#### Load dependencies
import tensorflow as tf
from tensorflow.keras import layers
from numpy import loadtxt, savetxt, reshape
import datetime as dt
print("TensorFlow Version: " + tf.__version__)
print("TF Keras Version: " + tf.keras.__version__)

#### Load data

start= dt.datetime.now() 
xTrain= loadtxt('Data/xTrain_Gesture0-6_980-32.csv')
yTrain= loadtxt('Data/yTrain_Gesture0-6_980-7.csv')

xTest= loadtxt('Data/xTest_Gesture0-6_420-32.csv')
yTest= loadtxt('Data/yTest_Gesture0-6_420-7.csv')

#Preprocess data (Normalization)

xTrain /= 75
xTest /= 75

n_classes = 7

# Design neural network architecture

model = tf.keras.Sequential() #One layer after the other
model.add(layers.Dense(20, activation='relu', input_shape=(32,))) #Dense= fully connected
model.add(layers.Dense(7, activation='softmax'))
model.summary()

# Configure model
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])

# Create a Callback for checking the model's value accuracy after each epoch
# If the accuracy has improved the model is saved (overwriten) for later use

filepath= 'Data/bestModel.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Training

#Adapt number of epochs to get eventually better results (recommended: 100 to 300)
model.fit(xTrain, yTrain, batch_size=32, epochs=600, verbose=0, validation_data=(xTest, yTest), callbacks=callbacks_list)
duration = (dt.datetime.now() - start)
print("\nDuration: " + str(duration))

# Load best model of training to evaluate it

model.load_weights(filepath)
# Compile model (required to evaluate the model)
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])

# Evaluation

model.evaluate(xTest, yTest)

# Access model

result= model.get_weights()
weights_h1 = result[0]
biases_b1= result[1]
weights_out= result[2]
biases_out= result[3]

# Save model files (csv)

print('\n--------------Saving model(csv)-------------')
print('Saving Data/NNweights_h1.csv')
savetxt('Data/NNweights_h1.csv', weights_h1, fmt='%10.8f', delimiter=' ')
print('Saving Data/NNbiases_b1.csv')
savetxt('Data/NNbiases_b1.csv', biases_b1, fmt='%10.8f', delimiter=' ')
print('Saving Data/NNweights_out.csv')
savetxt('Data/NNweights_out.csv', weights_out, fmt='%10.8f', delimiter=' ')
print('Saving Data/NNbiases_out.csv')
savetxt('Data/NNbiases_out.csv', biases_out, fmt='%10.8f', delimiter=' ')

# Save model files for Lego Mindstorms robot (rtf)

print('\n--Saving model (rtf for Lego Mindstorms EV3)--')
#Format: <number of rows>CR<number of columns>CR<{<aReal>CR}*
print('Saving Data/NNweights_h1.rtf')  
tmpArray = reshape(weights_h1, (weights_h1.shape[0] * weights_h1.shape[1],))
result= [weights_h1.shape[0],weights_h1.shape[1]] + tmpArray.tolist()
savetxt('Data/NNweights_h1.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')
     
print('Saving Data/NNbiases_b1.rtf')  
result= [1,biases_b1.shape[0]] + biases_b1.tolist()
savetxt('Data/NNbiases_b1.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Saving Data/NNweights_out.rtf')  
tmpArray = reshape(weights_out, (weights_out.shape[0] * weights_out.shape[1],))
result= [weights_out.shape[0],weights_out.shape[1]] + tmpArray.tolist()
savetxt('Data/NNweights_out.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Saving Data/NNbiases_out.rtf')  
result= [1,biases_out.shape[0]] + biases_out.tolist()
savetxt('Data/NNbiases_out.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Model saved.')