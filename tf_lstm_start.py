# Den nächsten Wert einer Reihe vorhersagen. 
# version: 1.0 10.3.2022

import numpy as np
import tensorflow as tf

trainInput  = np.asarray(
               [ [[0.01],[0.02],[0.03],[0.04],[0.05]],
                 [[0.02],[0.03],[0.04],[0.05],[0.06]],
                 [[0.03],[0.04],[0.05],[0.06],[0.07]],
                 [[0.04],[0.05],[0.06],[0.07],[0.08]],
                 [[0.05],[0.06],[0.07],[0.08],[0.09]],
                 [[0.06],[0.07],[0.08],[0.09],[0.1]],
                 [[0.07],[0.08],[0.09],[0.10],[0.11]],
                 [[0.08],[0.09],[0.10],[0.11],[0.12]],
                 [[0.09],[0.10],[0.11],[0.12],[0.13]],
                 [[0.10],[0.11],[0.12],[0.13],[0.14]] ] ).astype('float32')
trainOutput = np.asarray([0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]).astype('float32')

# model definieren
model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(5,1)))
model.add(tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_normal'))

model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model trainieren
model.fit(trainInput, trainOutput, epochs=200)

# model anwenden
testListe = np.asarray([ [[0.14], [0.15], [0.16], [0.17], [0.18]] ]).astype('float32')
vorhersage = model.predict(testListe)
print('Nächste Zahl: %.5f' % (vorhersage))
