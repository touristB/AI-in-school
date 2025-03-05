###############################################################################
# Überprüft ob typische Bibliotheken für KI/DL Entwicklung vorhanden sind
# author: A. Schindler ©
# since: 10.1.2018
# ver./ last edited: 15.8.2021
# This software comes without any warranty. 
###############################################################################

import os
import sys
import platform



print ("========================================================================================")
print ("||                                                                                    ||")
print ("||     KI-Bibliotheken checker - ver.: 15.8.2021                                      ||")
print ("||          Prüft ob typische Bibliotheken für KI/DL Entwicklung vorhanden sind.      ||")
print ("||                                                                                    ||")
print ("========================================================================================\n")


print ("\n==== OS ================================================================================")
print ("OS identification       ::", platform.platform())
print ("OS release              ::", platform.version())
print ("OS name                 ::", platform.system())
print ("sys.version             ::", sys.version.replace('\n', ''))
print ("Python ver.             ::",platform.python_version(), "         (tensorflow hat Probleme mit Python ver. 3.7)")
import struct
print ("Python                  :: ", end = '')
print(struct.calcsize("P") * 8, "- Bit        (tensorflow benötigt 64-Bit.)")
print ("Python Installationsort ::", os.path.dirname(sys.executable))




print ("\n==== Bibliotheken ======================================================================")

# KI
try:
    import tensorflow as tf
    print ("Bibliothek:: import tensorflow as tf                  ver.:", tf.__version__)
    print ("Bibliothek::        keras: tf.keras.__version__       ver.:", tf.keras.__version__)
except:
    print("ERROR: import tensorflow")



# für: https://github.com/arconsis/cifar-10-with-tensorflow2/blob/master/BetterNetwork.py
try:
    import tensorflow_datasets as tfds
    print ("Bibliothek:: import tensorflow_datasets as tfds       ver.:", tfds.__version__)
except:
    print("ERROR: import tensorflow_datasets")



# Für Regressionsbsp aus Tutorial
try:
    import tensorflow_docs as tfdocs
    #print (tfdocs.__doc__)
    print ("Bibliothek:: import tensorflow_docs as tfdocs         ver.: --OK")
except:
    print("ERROR: import tensorflow_docs")
    print("   install with:: pip install git+https://github.com/tensorflow/docs")



try:
    import tensorboard as tb
    print ("Bibliothek:: import tensorboard as tb                 ver.:", tb.__version__)
except:
    print("ERROR: import tensorboard")


try:
    import tf_agents as tfa
    print ("Bibliothek:: import tf_agents as tfa                  ver.:", tfa.__version__)
    #from tf_agents.agents.dqn import dqn_agent
except:
    print("ERROR: import tf_agents try: pip install tf-agents")




# für RL learning
try:
    import gymnasium as gym
    print("Bibliothek:: import gymnasium as gym                  ver.:", gym.__version__)
except:
    print("ERROR: import gymnasium")


try:
    import stable_baselines3 as sb3
    print("Bibliothek:: import stable_baselines3 as sb3          ver.:", sb3.__version__)
except:
    print("ERROR: stable_baselines")


# pytorch für stable_baselines3 
try:
    import torch as tor
    print("Bibliothek:: import torch as tor                      ver.:", tor.__version__)
except:
    print("ERROR: import torch")
    

try:
    import sklearn
    print ("Bibliothek:: import sklearn                           ver.:", sklearn.__version__)
except:
    print ("ERROR: import sklearn")

#import grpcio

# deprecated: keras ist Teil von Tensorflow
#try:
#    import keras
#    print ("Bibliothek:: import keras             ver.:", keras.__version__)
#except:
#    print ("ERROR: import keras")
#try:
    #from keras.models import Sequential
    #from keras.layers import Flatten, Dense, Activation
    #print ("Bibliothek:: from keras.models import Sequential")
    #print ("Bibliothek:: from keras.layers import Flatten, Dense, Activation")
#except:
    #print ("ERROR: import keras.models import Sequential")
    #print ("ERROR: from keras.layers import Flatten, Dense, Activation")




# Datenstrukturen
try:
    import pandas as pd
    print ("Bibliothek:: import pandas as pd                      ver.:", pd.__version__)
except:
    print ("ERROR: import pandas")

try:
    import seaborn as sns
    print ("Bibliothek:: import seaborn as sns                    ver.:", sns.__version__)
except:
    print ("ERROR: import seaborn")

try:
    import scipy.special
    print ("Bibliothek:: import scipy.special                     ver.:", scipy.__version__)
except:
    print ("ERROR: import scipy.special")

try:
    import numpy as np
    print ("Bibliothek:: import numpy                             ver.:", np.__version__)
except:
    print ("ERROR: import numpy")


# Anzeige
try:
    import matplotlib
    print ("Bibliothek:: import matplotlib                        ver.:", matplotlib.__version__)
except:
    print ("ERROR: import matplotlib")


try:
    import imageio
    print ("Bibliothek:: import imageio                           ver.:", imageio.__version__)
except:
    print ("ERROR: import imageio")


try:
    import glob
    print ("Bibliothek:: import glob                              ver.: -- OK")
except:
    print ("ERROR: import glob")


try:
    import IPython as ipt
    print ("Bibliothek:: import IPython as ipt                    ver.:", ipt.__version__)
except:
    print ("ERROR: import IPython")


try:
    from IPython.display import clear_output
    print ("Bibliothek:: from IPython.display import clear_output ver.: -- OK")
except:
    print ("ERROR: import IPython.display")


try:
    import PIL
    print ("Bibliothek:: import PIL                               ver.:", PIL.__version__, "    (Hinweis: PIL deprecated, Nachfolger:pillow) ")
except:
    print ("ERROR: import PIL (Hinweis: PIL deprecated, Nachfolger:pillow)")


try:
    # Pillow is the friendly PIL fork; Pillow and PIL cannot co-exist in the
    #same environment. Before installing Pillow, please uninstall PIL.
    import pillow
    print ("Bibliothek:: import pillow")
except:
    print ("ERROR: import pillow       (Hinweis: Parallelinstallation pillow u. PIL nicht möglich.")


#tf.keras.utils.plot_model( model, to_file='model.png')  # braucht pydot and graphviz
try:
    import pydot
    print ("Bibliothek:: import pydot")
except:
    print ("ERROR: import pydot")
try:
    import graphviz
    print ("Bibliothek:: import graphviz")
except:
    print ("ERROR: import graphviz")




# sonstiges:
try:
    import time as tm
    print ("Bibliothek:: import time                              ver.: -- OK")
except:
    print ("ERROR: import time")

try:
    import sqlite3
    print ("Bibliothek:: import sqlite3                           ver.:", sqlite3.version)
except:
    print ("ERROR: import sqlite3")

try:
    import langdetect
    print ("Bibliothek:: import langdetect                        ver.: -- OK")
except:
    print ("ERROR: import langdetect")

#################
print ("\n==== Pfade =============================================================================")
print("    Path at terminal when executing this file:: ", os.getcwd())

print("    This file path, relative to os.getcwd()  :: ", __file__ )

full_path = os.path.realpath(__file__)
print("    This file full path (following symlinks) :: ", full_path)

path, filename = os.path.split(full_path)
print("    This file directory and name             :: ", path + ' --> ' + filename)
print("    This file directory only                 :: ", os.path.dirname(full_path)+ "\n")
################

print ("==== Installation / update =============================================================")
print ("  Übersicht über verfügbare Module in Konsole: print(help('modules'))")
print ("  Conda::")
print ("      Bibliothek nachinstallieren als user:     ~/anaconda3/bin> ./conda install bibliotheksname ")
print ("      Bibliothek updaten als user               ~/anaconda3/bin> ./conda update tensorflow")
print ("      Bibliothek updaten unter Spyder console   In []: pip install tensorflow --upgrade")
print ("  sonstige Umgebung::")
print ("      Bibliothek installieren als admin > pip install bibliotheksname")
print ("      Bibliothek updaten als admin      > pip install tensorflow --upgrade")



print ("\n==== GPU prüfen ========================================================================")
print ("    tf.config.list_physical_devices('GPU') ::", tf.config.list_physical_devices('GPU') )
print ("    tf.test.gpu_device_name()              ::", tf.test.gpu_device_name())
print ("    Anzahl GPUs                            ::", len(tf.config.list_physical_devices('GPU')))


       
#import device_lib
#print(device_lib.list_local_devices())
#pip list | grep tensorflow


#import dummy
#print ("Bibliothek:: import dummy")

