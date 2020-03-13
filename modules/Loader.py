import pandas as pd
import numpy as np

class Loader:
  def __convertTo48Image(self, images):
    return np.array([np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in images.pixels])

  def __reshapeData(self, data):
    return data.reshape((-1, 48, 48, 1)).astype(np.float32)

  def __formatData(self, train,  validation, test):
    # Shape the pixels string of the image to a 48 by 48 pixel image.
    return self.__reshapeData(self.__convertTo48Image(train)), self.__reshapeData(self.__convertTo48Image(validation)), self.__reshapeData(self.__convertTo48Image(test))

  def __getDataByUsages(self, dataset):
    trainData = dataset[dataset['Usage'] == 'Training']
    validationData = dataset[dataset['Usage'] == 'PublicTest']
    testData = dataset[dataset['Usage'] == 'PrivateTest']

    return trainData, validationData, testData

  def loadFEC_dataset(self, location='../datasets/icml_face_data.csv'):
    dataset = pd.read_csv('./datasets/icml_face_data.csv', sep=r'\s*,\s*', engine='python')

    # Filter to separate dataset by Usage.
    trainData, validationData, testData = self.__getDataByUsages(dataset)

    # Assign type to label.
    trainLabel = trainData.emotion.astype(np.int32).values
    validLabel = validationData.emotion.astype(np.int32).values
    testLabel = testData.emotion.astype(np.int32).values

    trainData, validationData, testData = self.__formatData(trainData, validationData, testData)

    # Normalize data based on the max amount for a pixel.
    trainDataNormalized = trainData / 255.0
    validationDataNormalized = validationData / 255.0
    testDataNormalized = testData / 255.0

    return (
      (trainDataNormalized, trainLabel),
      (validationDataNormalized, validLabel),
      (testDataNormalized, testLabel)
    )