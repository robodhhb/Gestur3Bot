# Gestur3Bot
Software and construction manual for the Gestur3Bot Lego Mindstorms EV3 robot

Willkommen beim Gestur3Bot Roboter! (see English version below)

Dieser Roboter kann mit 7 Gesten gesteuert werden, die ein Benutzer vor den zwei Infrarotsensoren ausführt. 
Das Video "Gestur3Bot SD" zeigt, wie ein Benutzer mit der Faust den Roboter steuert. 
Die Gesten erkennt der Roboter mit Hilfe eines neuronales Netzes, das zuvor mit Trainingsdaten und einem Trainingsprogramm
auf Basis von Keras und TensorFlow auf einem PC trainiert wurde. Der Merkmalsvektor besteht aus 32 
Entferneungswerten, je 16 pro IR-Sensor.  Das neuronale Netz hat ein versteckte Schicht von
20 Neuronen und eine Ausgsbeschicht von 7 Neuronen. Pro Klasse (Geste) bestehen die Trainingsdaten aus 
140 und die Testdaten aus 60 Merkmalsvektoren. 
Im Repository befindet sich ein Modell mit 94,8% Klassifikationsgüte. Für weitere Informationen 
zum Bau und Training des Roboters siehe die ersten beiden Ordner in diesem Repository. Das Projekt ist ausführlich beschrieben im deutschen Magazin "Make" in der Ausgabe 6/2018.

Welcome to the Gestur3Bot robot!

This robot can be controlled by 7 gestures, which a user performs in front of the two infra-red sensors
of the robot. The video "Gestur3Bot SD" shows how a user controls the robot with a fist. The robot recognizes
a gesture by using a neural network which has been trained with training data and a training program based on Keras and
Tensorflow on a PC. The feature vector contains 32 distance values, 16 coming from each sensor.
The neural network consists of one hidden layer with 20 neurons and one output layer with 7 neurons.
The training phase uses 140 feature vectors for each class (gesture) for the training data
and 60 feature vectors for each class for the test data. 
You can find a model with 94,8% validation accuracy 
in this repository. Please see the first two folders of this repository for further information
for the construction and training of the robot. This project is described in the issue 6/2018 of the German 
Make magazine.

