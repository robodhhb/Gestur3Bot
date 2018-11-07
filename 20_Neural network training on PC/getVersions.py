#####################################################
# Display versions of some installed packages
# 
# Author: Detlef Heinze 
# Version: 1.0   Date: 13.09.2018
#####################################################
import tensorflow as tf  
import keras as k

#Abschalten der Warnmeldungen von TensorFlow, die
#melden, dass die TensorFlow-Library nicht 
#alle Eigenschaften dieser HW nutzt.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print()
print('TensorFlow Version: ' + tf.__version__)
print('Keras Version: ' + k.__version__)
