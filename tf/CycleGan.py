from keras.models import Model,Sequential
from keras.layers import Dense, Input, Flatten
from keras.layers import BatchNormalization,Activation,Reshape,LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose,UpSampling2D
from keras.optimizers import RMSprop
import numpy as np
import os

class CycleGan(object):
    def __init__(self, img_height, img_width, img_channel):
        self.FG = None
        self.BG = None
        self.D = None
        self.CGAN = None
        self.AM = None
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel

    def _createGenerater(self,img_height, img_width, img_channel):
        depth = 64
        Generator = Sequential()
        Generator.add(Conv2D(depth * 1, 5, input_shape=(img_height, img_width, img_channel), padding='same'))
        Generator.add(Activation('relu'))
        Generator.add(MaxPooling2D())
        Generator.add(Conv2D(depth * 2, 5, padding='same'))
        Generator.add(Activation('relu'))
        Generator.add(MaxPooling2D())
        Generator.add(Conv2D(depth * 4, 5, padding='same'))
        Generator.add(Activation('relu'))
        Generator.add(MaxPooling2D())
        Generator.add(Conv2D(depth * 8, 5, padding='same'))
        Generator.add(Activation('relu'))
        Generator.add(MaxPooling2D())
        Generator.add(Conv2DTranspose(int(depth * 4), 5, padding='same'))
        #Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(int(depth * 2), 5, padding='same'))
        #Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(int(depth * 1), 5, padding='same'))
        #Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(img_channel, 5, padding='same'))
        Generator.add(Activation('sigmoid'))
        Generator.add(UpSampling2D())
        Generator.summary()
        return Generator

    def _createDiscriminator(self,img_height,img_width,img_channel):
        depth = 64
        Discriminator = Sequential()
        Discriminator.add(Conv2D(depth * 1, 5, input_shape=(img_height, img_width, img_channel), padding='same'))
        Discriminator.add(Activation('relu'))
        Discriminator.add(MaxPooling2D())
        Discriminator.add(Conv2D(depth * 2, 5, padding='same'))
        Discriminator.add(Activation('relu'))
        Discriminator.add(MaxPooling2D())
        Discriminator.add(Conv2D(depth * 4, 5, padding='same'))
        Discriminator.add(Activation('relu'))
        Discriminator.add(MaxPooling2D())
        Discriminator.add(Conv2D(depth * 8, 5, padding='same'))
        Discriminator.add(Activation('relu'))
        Discriminator.add(MaxPooling2D())
        Discriminator.add(Flatten())
        Discriminator.add(Dense(1))
        Discriminator.add(Activation('sigmoid'))
        Discriminator.summary()

        return Discriminator

    def generaterModel(self):
        if self.FG:
            return self.FG
        optimizer = RMSprop(lr=0.002, decay=6e-8)
        gen=self._createGenerater(self.img_height, self.img_width, self.img_channel)
        self.FG = Sequential()
        self.FG.add(gen)
        self.FG.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.FG

    def discriminator(self):
        if self.D:
            return self.D
        optimizer = RMSprop(lr=0.002, decay=6e-8)
        dis=self._createDiscriminator(self.img_height, self.img_width, self.img_channel)
        self.D = Sequential()
        self.D.add(dis)
        self.D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.D

    def backwardGeneraterModel(self):
        if self.BG:
            return self.BG
        optimizer = RMSprop(lr=0.002, decay=6e-8)
        self.BG = Sequential()
        self.BG.add(self._createGenerater(self.img_height, self.img_width, self.img_channel))
        self.BG.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.BG

    def cycleModel(self):
        if self.CGAN:
            return self.CGAN
        optimizer = RMSprop(lr=0.002, decay=6e-8)
        self.CGAN = Sequential()
        self.CGAN.add(self.generaterModel())
        self.CGAN.add(self.backwardGeneraterModel())
        self.CGAN.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.CGAN

    def advarseModel(self):
        if self.AM:
            return self.AM

        gn=self.generaterModel()
        dm = self.discriminator()
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        dm.trainable = False
        self.AM = Sequential()
        self.AM.add(gn)
        self.AM.add(dm)
        self.AM.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.AM.summary()
        dm.trainable = True
        return self.AM

    def train_on_batch(self, datasetX, datasetY):
        gen=self.generaterModel()
        dis=self.discriminator()
        cgan=self.cycleModel()
        adv=self.advarseModel()

        disLoss=0
        disAcc=0
        amLoss=0
        amAcc=0
        cygLoss=0
        cygAcc=0

        #train discriminator
        generatedY = gen.predict(datasetX)
        y = np.zeros([ datasetX.shape[ 0 ], 1 ])
        disLoss, disAcc = dis.train_on_batch(generatedY, y)

        y = np.ones([datasetX.shape[0], 1])
        #disLoss1=0
        #disAcc1=0
        disLoss1, disAcc1=dis.train_on_batch(datasetY, y)



        disLoss=(disLoss+disLoss1)/2
        disAcc = (disAcc + disAcc1) / 2


        #train the generater
        y = np.ones([datasetX.shape[0], 1])
        amLoss,amAcc=adv.train_on_batch(datasetX,y)

        #train the cycleGan (minimize cycleloss)
        cygLoss,cygAcc=cgan.train_on_batch(datasetX,datasetY)

        return disLoss,disAcc,amLoss,amAcc,cygLoss,cygAcc

    def saveModel(self, path):
        if not os.path.exists(path=path):
            os.mkdir(path)
        self.FG.save(path+"/fgen.h5")
        self.BG.save(path+"/bgen.h5")
        self.D.save(path+"/dis.h5")

    def loadModel(self, path):
        gen=self.generaterModel()
        bgen=self.backwardGeneraterModel()
        dis=self.discriminator()
        if(os.path.exists(path=path+"/fgen.h5")):
            gen.load_weights(path+"/fgen.h5")
            bgen.load_weights(path+"/bgen.h5")
            dis.load_weights(path+"/dis.h5")