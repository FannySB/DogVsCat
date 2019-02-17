"""
Made by: 
Vincent Morissette-Thomas
Fanny Salvail-Bérard
Vilayphone Vilaysouk

Inspired by:
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import callbacks
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import os.path
import math
import matplotlib.pyplot as plt


def plot_training_history(history2):
  # Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(history2.history['loss'],'r',linewidth=2.0)
  plt.plot(history2.history['val_loss'],'b',linewidth=2.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=20, fontweight='bold')
  plt.savefig("Loss.png")

  # Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(history2.history['acc'],'r',linewidth=2.0)
  plt.plot(history2.history['val_acc'],'b',linewidth=2.0)
  plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=20, fontweight='bold')
  plt.savefig("Accuracy.png")


# dimensions of our images.
# img_width, img_height = 150, 150
img_width, img_height = 64, 64

train_data_dir = 'dataset/trainset/train'
validation_data_dir = 'dataset/trainset/valid/'
test_data_dir = 'dataset/testset/'
nb_train_samples = 17998 # of 17998 total
nb_validation_samples = 2000 # of 2000 total
nb_test_samples = 4999 # of 4999 total
epochs = 100
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)




with tf.device('/gpu:0'):
    def create_model():

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

        return model

model = create_model()


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# model.evaluate_generator(generator=validation_generator)

test_generator.reset()
pred = model.predict_generator(test_generator, steps=math.ceil(nb_test_samples/batch_size), verbose=1)
pred_indices = [1 if x>0.5 else 0 for x in pred]
labels = (train_generator.class_indices)
labels = dict((v, k) for k,v in labels.items())

predictions = [labels[k] for k in pred_indices]

filenames = test_generator.filenames
filepath = [x.split("/") for x in filenames]
id = [x[1].split(".")[0] for x in filepath]
results = pd.DataFrame({"id": id, "label": predictions})
results.to_csv("submission.csv", index=False)

plot_training_history(history)

model.save_weights('first_try.h5')