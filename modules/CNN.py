import tensorflow as tf
import matplotlib.pyplot as plt

class CNN:
  def __init__(self):
    self.model = self.__createModel()
    self.history = None

  def plotEvaluationModel(self):
    plt.plot(self.history.history['accuracy'], label='Accuracy')
    plt.plot(self.history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1])
    plt.legend(loc='lower right')
    plt.show()

  def __createLayersCNN(self, model):
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Output layer -> 7 possible emotion output.
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

  def __createModel(self):
    model = tf.keras.models.Sequential()
    self.__createLayersCNN(model)
    return model

  def displayArchitecture(self):
    self.model.summary()

  def fit(self, dataset, target, validationData, validationTarget, **kwargs):
    self.model.compile(loss="sparse_categorical_crossentropy", optimizer='adam' , metrics=['accuracy'])
    self.history = self.model.fit(dataset, target,
      validation_data=(validationData, validationTarget),
      shuffle=True,
      **kwargs
    )
    # self.model.evaluate(test_images,  test_labels, verbose=2)

  def predict(self, dataset):
    pass