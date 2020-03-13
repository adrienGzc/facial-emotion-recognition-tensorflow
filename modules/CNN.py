from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np

class CNN:
  def __init__(self):
    self.__model = self.__createModel()
    self.__history = None
    self.__saveFolder = './model'

  # def plotEvaluationModel(self):
  #   plt.plot(self.__history.history['accuracy'], label='Accuracy')
  #   plt.plot(self.__history.history['val_accuracy'], label = 'Validation Accuracy')
  #   plt.xlabel('Epoch')
  #   plt.ylabel('Accuracy')
  #   plt.ylim([0.2, 1])
  #   plt.legend(loc='lower right')
  #   plt.show()

  def __createLayersCNN(self, model):
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Output layer -> 7 possible emotion output.
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

  def __createModel(self):
    model = tf.keras.models.Sequential()
    self.__createLayersCNN(model)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam' , metrics=['accuracy'])
    return model

  def showArchitecture(self):
    self.__model.summary()

  def getModel(self):
    return self.__model

  def __saveModelCallback(self):
    return tf.keras.callbacks.ModelCheckpoint(
      filepath='./model/',
      verbose=1
    )

  def __loadExistingModel(self):
    try:
      self.__model = tf.keras.models.load_model('./model/')
      print('Info: Existing model found. If you want to force the fitting use this parameter forceFit=True.')
      return True
    except:
      print('Info: No model found. Start create a new one...')
      return False

  def fit(self, dataset, target, validationData, validationTarget, save=True, forceFit=False, **kwargs):
    if forceFit is True or self.__loadExistingModel() is False:
    # The callback saved the model after each epoch in case of interuption.
      self.__model.fit(dataset, target,
        validation_data=(validationData, validationTarget),
        shuffle=True,
        callbacks=[self.__saveModelCallback()],
        **kwargs
      )

      # Save entire model at the end of the training.
      if save is True:
        self.__model.save(filepath='./model/', save_format='tf', overwrite=True, include_optimizer=True)

  def getAccuracy(self, predictions, label):
    acc = 0
    for index, predict in enumerate(predictions):
      if np.argmax(predict) == label[index]:
        acc += 1

    if acc == 0:
      return acc
    return (acc / len(predictions)) * 100

  def predict(self, dataset, target=None):
    return self.__model.predict(dataset)
