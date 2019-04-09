import torch

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import Sequential
from keras import layers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

x = torch.load('features.pth').data.numpy()
y = torch.load('labels.pth').data.numpy()

x = x[:400]
y = y[:400]

model = Sequential()
model.add(layers.Dense(2048, activation='relu', input_dim=x.shape[1]))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=SGD(0.01, momentum=0.9),
              metrics=['accuracy'])

model.fit(x, y, validation_split=0.1, epochs=100, batch_size=64,
          callbacks=[ReduceLROnPlateau(factor=0.5, verbose=1)])
model.save('clf.hdf5')
