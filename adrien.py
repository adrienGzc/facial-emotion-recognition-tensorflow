import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import cv2

from modules import CNN, Loader

def getFilePath(file, folder):
  pwd = os.path.abspath(folder) + '/' + os.path.dirname(file)
  return pwd + file

def reshapeStringPixel(pixel):
  return np.asarray(list(pixel.split(' ')), dtype=np.uint8).reshape((48, 48))

def createImageFromPixel(folder, dataset, limit=None):
  pwd = os.getcwd()
  imagesFolder = pwd + '/' + folder

  if not os.path.exists(imagesFolder):
    os.makedirs(imagesFolder)
  for index, row in dataset.iterrows():
    if limit is not None and index + 1 > limit:
      break
    pixels = reshapeStringPixel(row['pixels'])
    pathname = os.path.join(imagesFolder, str(index + 1) + '.jpg')
    cv2.imwrite(pathname, pixels)
    print('Image saved: {}'.format(pathname))

def imageToPixel(imageFile):  
  return cv2.imread(imageFile)

def convertImagesToPixels(folder):
  listImages = os.listdir(folder)
  images = list()

  for image in listImages:
    pathImage = getFilePath(image, folder)
    images.append(imageToPixel(pathImage))
  return np.asarray(images)

def showImage(image):
  plt.imshow(image)
  plt.show

def main():
  cnn = CNN.CNN()
  loader = Loader.Loader()
  (trainData, trainLabel), (validationData, validationLabel), (_testData, _testLabel) = loader.loadFEC_dataset()
  cnn.fit(trainData, trainLabel, validationData, validationLabel, epochs=10, batch_size=64)
  cnn.plotEvaluationModel()

if __name__ == "__main__":
    main()