# 21.9.2020
# A. Schindler

import tensorflow as tf                      

# Erstellen der Trainingsdaten
inputMuster = [1, 2, 4] 
outputMuster= [3, 6, 12 ] 

# Aufbau des neuronalen Netzwerkes
model = tf.keras.Sequential()     
model.add(tf.keras.layers.Dense(1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# trainieren des neuronalen Netzwerkes
model.fit(inputMuster, outputMuster, epochs=1000)


# Testen des neuronalen Netzwerkes mit Testdaten
testMuster =  [22]
print(model.predict(testMuster))

