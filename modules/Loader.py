import pandas as pd
import numpy as np

class Loader:
  def __init__(self):
    pass

  def loadFEC_dataset(self, location='../datasets/icml_face_data.csv'):
    df = pd.read_csv('./datasets/icml_face_data.csv', sep=r'\s*,\s*', engine='python')

    # Filter to separate dataset by Usage.
    train_samples = df[df['Usage'] == 'Training']
    validation_samples = df[df['Usage'] == 'PublicTest']
    test_samples = df[df['Usage'] == 'PrivateTest']

    # Assign type to label.
    y_train = train_samples.emotion.astype(np.int32).values
    y_valid = validation_samples.emotion.astype(np.int32).values
    y_test = test_samples.emotion.astype(np.int32).values

    # Shape the pixels string of the image to a 48 by 48 pixel image.
    X_train = np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in train_samples.pixels])
    X_valid = np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in validation_samples.pixels])
    X_test = np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in test_samples.pixels])

    X_train = X_train.reshape((-1, 48, 48, 1)).astype(np.float32)
    X_valid = X_valid.reshape((-1, 48, 48, 1)).astype(np.float32)
    X_test = X_test.reshape((-1, 48, 48, 1)).astype(np.float32)

    # Normalize data.
    X_train_std = X_train / 255.
    X_valid_std = X_valid / 255.
    X_test_std = X_test / 255.

    return (X_train_std, y_train), (X_valid_std, y_valid), (X_test_std, y_test)