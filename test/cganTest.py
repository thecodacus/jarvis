from tf.CycleGan import CycleGan
from utils.DataLoader import DataLoader
import numpy as np

imageShape=(32,32,3)

dl=DataLoader(path="../dataset",batchSize=10,imageSize=imageShape[0])

cgan=CycleGan(imageShape[0], imageShape[1], imageShape[2])

epochs=10
modelSavePath='cgan_saved'
cgan.loadModel(modelSavePath)
for i in range(epochs):
    dataset = dl.getGenerater()
    for data in dataset:
        datasetX = np.array(data[0])
        datasetY = np.array(data[1])
        report=cgan.train_on_batch(datasetX=datasetX,datasetY=datasetY)
        print(report)

    cgan.saveModel(modelSavePath)

