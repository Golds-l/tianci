from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

train_img_path = "data/train"
val_img_path = "data/validation"

train_img_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   fill_mode="nearest")

train_img = train_img_gen.flow_from_directory(directory=train_img_path,
                                              target_size=(32, 64),
                                              shuffle=True,
                                              batch_size=40,
                                              class_mode="categorical")

train_img_num = train_img.n
# print(train_img_num)

val_img_gen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=20,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 fill_mode="nearest"
                                 )

val_img = val_img_gen.flow_from_directory(directory=val_img_path,
                                          batch_size=40,
                                          shuffle=False,
                                          target_size=(32, 64),
                                          class_mode="categorical")
val_img_num = val_img.n

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(x=train_img, epochs=40, validation_data=val_img)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


