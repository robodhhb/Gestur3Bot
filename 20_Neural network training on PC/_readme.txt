Tools und Daten für das neuronale Netzwerk des Gestur3Bot Lego Mindstorms EV3 roboter
(see English version below)

Voraussetzungen auf dem PC
1. Ein 64bit Windows PC mit Windows 7 oder höher.

2. Python 3.6.3 (64 bit Version) (Click "Add Python 3.6 to path" im installation dialog)
   Download-Adresse: 
   https://www.python.org/ftp/python/3.6.3/python-3.6.3-amd64.exe
   
3. Library "TensorFlow" (1.8.0 oder höher)
   Installation mit der Eingabeaufforderung:
   pip3 install --upgrade tensorflow
   Das Projekt ist mit der TensorFlow-Version 1.12.0 erfolgreich getestet.

4. Library KERAS (2.1.3 oder höher)
   pip3 install keras 

5. Fileformat h5py für Model checkpoints in Keras
   pip3 install h5py
   
   
Das Training des neuronalen Netzes kann mit jeweils einem der folgenden 
Programme auf dem PC erfolgen: (Voraussetzung: Unterverzeichnis: /Data mit Trainings- u. Testdaten
                                und /logs vorhanden)
 
 - train_NN_Keras.py             (Implementierung mit KERAS auf Basis von TensorFlow)
 - train_NN_Keras_Tensorboard.py (Mitschrift des Trainingsverlaufes mittels Tensorboard
                                  Siehe Hinweise im Programmkopf) 
 - train_NN_Keras_opt.py         (Optimierte Implementierung, die das beste Modell eines Laufs festhält)
 
Alle obigen Programme lesen folgende Dateinen aus dem /Data Verzeichnis:
   xTrain_Gesture0-6_980-32.csv
   yTrain_Gesture0-6_980-7.csv
   xTest_Gesture0-6_420-32.csv
   yTest_Gesture0-6_420-7.csv
   
Starten eines der obigen Programme:
1) Eingabeaufforderung öffnen
2) Wechslen Sie in den Ordner, wo die Programme auf ihrem PC liegen
3) Starten Sie eines der Programme mit:   python <programName>.py

Folgende Modelldateien werden im Verzeichnis /Data erzeugt: 
   NNbiases_b1.csv
   NNbiases_out.csv
   NNweights_h1.csv
   NNweights_out.csv
sowie die Modelldateien für den Lego Mindstorm EV3 roboter
   NNbiases_b1.rtf
   NNbiases_out.rtf
   NNweights_h1.rtf
   NNweights_out.rtf
Vorhandene Dateien werden überschrieben.

Installation der Modelldateien auf dem Gestur3Bot EV 3 Lego Roboter: 
Verwenden sie den Speicher-Browser der Lego-Entwicklungsumgebung, um die 
4 rtf-Dateien in das Projektverzeichnis der Gestur3Bot-Anwendung zu kopieren.
(Hinweis: Projektverzeichnis selektieren)

Utilities:
getVersion.py zeigt die installierten Versionen von TensorFlow und Keras an.


-------------------   
--English version--
-------------------
Tools and data for the neural network of the Gestur3Bot Lego Mindstorms EV3 roboter. 

Prerequisites on the PC
1. A 64bit Windows PC with Windows 7 or higher.

2. Python 3.6.3 (64 bit Version) (Click "Add Python 3.6 to path" on the installation dialog)
   Download-Adresse: 
   https://www.python.org/ftp/python/3.6.3/python-3.6.3-amd64.exe
   
3. Library "TensorFlow" (1.8.0 or higher)
   Installation with a command promot:
   pip3 install --upgrade tensorflow
   This project is successfully tested with TensorFlow Version 1.12.0

4. Library KERAS (2.1.3 or higher)
   pip3 install keras 

5. Fileformat h5py used for Model checkpoints in Keras
   pip3 install h5py
   
   
The training can be executed with one of the following programs on PC.
Prerequisites: A /Data subdirectory with the training- and testdata files
               and an empty /logs subdirectory.
 - train_NN_Keras.py             (Implementation with KERAS on basis of TensorFlow)
 - train_NN_Keras_Tensorboard.py (Implementation with KERAS using Tensorboard additionally) 
 - train_NN_Keras_opt.py         (Optimized Implementation, which saves the best model during training)
   
 All programs above read the following files from the /Data subdirectory:
   xTrain_Gesture0-6_980-32.csv
   yTrain_Gesture0-6_980-7.csv
   xTest_Gesture0-6_420-32.csv
   yTest_Gesture0-6_420-7.csv

How to run one of the three programs:
1) Open a command prompt
2) Change directory to the folder where the three programms are stored
3) Execute one of the three programs with:  python <programName>.py

The resulting model files are stored in /Data: 
   NNbiases_b1.csv
   NNbiases_out.csv
   NNweights_h1.csv
   NNweights_out.csv
as well as the model files for the  Lego Mindstorm EV3 roboter
   NNbiases_b1.rtf
   NNbiases_out.rtf
   NNweights_h1.rtf
   NNweights_out.rtf
Existing files are deleted.

Installation of the model files on the Gestur3Bot EV 3 Lego roboter: 
Use the memory-browser to put the 4 rtf-Files into the project directory of
the Gestur3Bot_V1 application. 
(Hint: Select project directory)

Utilities:
getVersion.py prints the installed versions of TensorFlow and Keras.
