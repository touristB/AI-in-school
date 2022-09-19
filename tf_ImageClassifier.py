# Bilder klassifizieren 
# last edited: 11.09.2021  version: 1.1 author: A. Schindler
############################################################################
from PIL import Image
import tensorflow as tf
import numpy as np

### Daten vorbereiten ###
trainBilder = tf.keras.preprocessing.image_dataset_from_directory('data/train', image_size=(200,200))

### Klassennamen ausgeben: 
anzahlKlassen = len(trainBilder.class_names)
print("Folgende", anzahlKlassen, "Klassen wurden erkannt:", trainBilder.class_names)


### Model erstellen ###
model = tf.keras.Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(anzahlKlassen))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


### Model trainieren ###
model.fit(trainBilder, epochs=10)


### Model testen ###
img = tf.keras.preprocessing.image.load_img('data/imageTestDand.jpg', target_size=(200, 200)) # gibt PIL image zurück
img_array = tf.keras.preprocessing.image.img_to_array(img)                                    # Converts a PIL Image instance to a Numpy array.    
img_array = np.array([img_array])                                                             # convert single image to batch
predictions = model.predict(img_array) 
score = tf.nn.softmax(predictions[0]) 

### Ausgabe 
Image.open('data/imageTestDand.jpg').show()
print("Mit einer Wahrscheinlichkeit von {:.2f} % ist das Bild ein: {}".format(100 * np.max(score), trainBilder.class_names[np.argmax(score)]))
