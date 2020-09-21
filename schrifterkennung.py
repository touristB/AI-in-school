# 21.9.2020
# A. Schindler


import tensorflow as tf
import numpy
import matplotlib
# 1. Daten aufbereiten
mnist = tf.keras.datasets.mnist

(trainZiffernBilder, trainZiffernLabels),(testZiffernBilder, testZiffernLabels) = mnist.load_data() 
trainZiffernBilder = trainZiffernBilder / 255.0
testZiffernBilder = testZiffernBilder / 255.0

# 2. Model erstellen
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Model trainieren
model.fit(trainZiffernBilder, trainZiffernLabels, epochs=5)
print("4. Model prüfen")
modWeighsBsp3 = model.get_weights()

# 4. Model prüfen
verlust, genauigkeit = model.evaluate(testZiffernBilder, testZiffernLabels)
print('Verlust: ', verlust, 'Genauigkeit: ', genauigkeit)

# 5. Testen des neuronalen Netzwerkes mit Testdaten
gesuchteZahlIndex = 0
erkennungsRaten = model.predict (testZiffernBilder [gesuchteZahlIndex:gesuchteZahlIndex+1])

flattendEr = erkennungsRaten.flatten()
flattendTZLabels = testZiffernLabels.flatten()

# 6. Test Ausgabe der gesuchten Zahl als Bild
imageArray = numpy.asfarray (testZiffernBilder [gesuchteZahlIndex:gesuchteZahlIndex+1]).reshape((28,28))
matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
print ("gesuchte Zahl: ", flattendTZLabels[gesuchteZahlIndex])

# Ausgabe der Erkennungsraten für die Zahlen 0..9
counter = 0
while counter < 10:
    readAbleErkennung = flattendEr[counter] * 10000
    readAbleErkennung = readAbleErkennung.astype(int)
    readAbleErkennung = readAbleErkennung / 10000
    print (" Zahl:", counter, "  Erkennungsrate:", readAbleErkennung)
    counter = counter + 1
