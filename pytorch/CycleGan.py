import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleGan(nn.Module):
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
        Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(int(depth * 2), 5, padding='same'))
        Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(int(depth * 1), 5, padding='same'))
        Generator.add(BatchNormalization(momentum=0.9))
        Generator.add(Activation('relu'))
        Generator.add(UpSampling2D())
        Generator.add(Conv2DTranspose(img_channel, 5, padding='same'))
        Generator.add(Activation('sigmoid'))
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
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.FG = Sequential()
        self.FG.add(self._createGenerater(self.img_height, self.img_width, self.img_channel))
        self.FG.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.FG

    def discriminator(self):
        if self.D:
            return self.D
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.D = Sequential()
        self.D.add(self._createDiscriminator(self.img_height, self.img_width, self.img_channel))
        self.D.compile(loss='mse', optimizer=optimizer, metrics=[ 'accuracy' ])
        return self.D

    def backwardGeneraterModel(self):
        if self.BG:
            return self.BG
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.BG = Sequential()
        self.BG.add(self._createGenerater(self.img_height, self.img_width, self.img_channel))
        self.BG.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.BG

    def cycleModel(self):
        if self.CGAN:
            return self.CGAN
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.CGAN = Sequential()
        self.CGAN.add(self.generaterModel())
        self.CGAN.add(self.backwardGeneraterModel())
        self.CGAN.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.CGAN

    def advarseModel(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.AM = Sequential()
        self.AM.add(self.generaterModel())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return self.AM