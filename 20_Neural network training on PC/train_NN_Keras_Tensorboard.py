#####################################################
# Trainingsprogramm für das neuronale Netz des
# GestureBot Roboters mit Hilfe von KERAS und
# Tensorflow.  Die Architektur des NN ist 32-20-7
# Der Adam-Optimizer wird verwendet. 
# Tensorboard wird verwendet, um den Verlauf des
# Lernvorganges anhand der accuracy zu zeigen. 
# Nachdem dieses Programm gelaufen ist Tensorboard 
# wie folgt starten: Im directory, wo dieses Program
# liegt: tensorboard --logdir=./logs/ 
# Danach mit einem Browser (nicht IE11 !) auf
# http://yourMachine:6006 zugreifen

# Autor: Detlef Heinze 
# Version: 1.1      Datum: 07.06.2018       
#####################################################
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from numpy import loadtxt, savetxt, reshape
import datetime as dt

#Abschalten der Warnmeldungen von TensorFlow, die
#melden, dass die TensorFlow-Library nicht 
#alle Eigenschaften dieser HW nutzt.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Training eines Neuronalen Netzes mit KERAS und TensorBoard(V1.1)\n')
start= dt.datetime.now()

# Step 1: Import Training Data (xTrain und yTrain)
print('Lese xTrain- und yTrain-Daten')
xTrain= loadtxt('Data/xTrain_Gesture0-6_980-32.csv')
yTrain= loadtxt('Data/yTrain_Gesture0-6_980-7.csv')

xTest= loadtxt('Data/xTest_Gesture0-6_420-32.csv')
yTest= loadtxt('Data/yTest_Gesture0-6_420-7.csv')

# Step 2: Normalisierung
xTrain /= 75
xTest /= 75

# Step 3: Definition der neuronalen Netzes
# Parameter für das 32 - 20 - 7 neuronale Netz
n_input = 32   #32 Neuronen für die gemessene Daten
n_hidden_1 = 20 #Größe der versteckten Neuronenschicht
n_classes = 7  #Größe der ausgebenden Neuronenschicht
               #ist gleich der Anzahl der Klassen
               
model = Sequential() #Ein Layer nach dem anderen
model.add(Dense(n_hidden_1, activation='relu', input_shape=(n_input,))) #Dense= fully connected
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

#Tensorboard konfigurieren
tensorbrd = TensorBoard('logs/GestureBot')
tensorbrd.log_dir

#Step 4: Training
# Die Epochenanzahl kann angepasst werden, um bessere Ergebnisse zu bekommen.
# Empfohlen: Werte von 100 bis 500
model.fit(xTrain, yTrain, batch_size=1, epochs=100, verbose=2, 
          validation_data=(xTest, yTest), callbacks=[tensorbrd])
duration = (dt.datetime.now() - start)
print("\nDauer: " + str(duration))

#Step 5: Evaluierung
print('Klassifikationgüte: ' + repr(model.evaluate(xTest, yTest)[1]))

#Step 6: Sicherung des berechneten Modells als csv und rtf
#Ermittlung der weights und biases
result= model.get_weights()
weights_h1 = result[0]
biases_b1= result[1]
weights_out= result[2]
biases_out= result[3]

print('\n--------------Model wird gespeichert(csv)-------------')
print('Sichere Data/NNweights_h1.csv')
savetxt('Data/NNweights_h1.csv', weights_h1, fmt='%10.8f', delimiter=' ')
print('Sichere Data/NNbiases_b1.csv')
savetxt('Data/NNbiases_b1.csv', biases_b1, fmt='%10.8f', delimiter=' ')
print('Sichere Data/NNweights_out.csv')
savetxt('Data/NNweights_out.csv', weights_out, fmt='%10.8f', delimiter=' ')
print('Sichere Data/NNbiases_out.csv')
savetxt('Data/NNbiases_out.csv', biases_out, fmt='%10.8f', delimiter=' ')

print('\n--Model wird gespeichert(rtf for Lego Mindstorms EV3)--')
#Format: <number of rows>CR<number of columns>CR<{<aReal>CR}*
print('Sichere Data/NNweights_h1.rtf')  
tmpArray = reshape(weights_h1, (weights_h1.shape[0] * weights_h1.shape[1],))
result= [weights_h1.shape[0],weights_h1.shape[1]] + tmpArray.tolist()
savetxt('Data/NNweights_h1.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')
     
print('Sichere Data/NNbiases_b1.rtf')  
result= [1,biases_b1.shape[0]] + biases_b1.tolist()
savetxt('Data/NNbiases_b1.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Sichere Data/NNweights_out.rtf')  
tmpArray = reshape(weights_out, (weights_out.shape[0] * weights_out.shape[1],))
result= [weights_out.shape[0],weights_out.shape[1]] + tmpArray.tolist()
savetxt('Data/NNweights_out.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Sichere Data/NNbiases_out.rtf')  
result= [1,biases_out.shape[0]] + biases_out.tolist()
savetxt('Data/NNbiases_out.rtf', result, fmt='%10.8f', delimiter='\r', newline='\r')

print('Model gesichert.')

