import os
import cv2


class DataLoader(object):
    def __init__(self,path,batchSize=30, imageSize=112):
        self.dataPaths = {}
        self.dataClasses = []
        self.rootPath=path
        self.batchSize=batchSize
        self.imageSize = imageSize
        for _, dirs, _ in os.walk(path):
            self.dataClasses.extend(dirs)
            break

        self.minLen=None
        for class_ in self.dataClasses:
            self.dataPaths[class_]=[]
            for root, _, filenames in os.walk(os.path.join(self.rootPath,class_)):
                filenames = [k for k in filenames if k.split(".")[-1] in "jpeg jpg png"]
                self.dataPaths[class_].extend(filenames)
                if self.minLen is None:
                    self.minLen = len(self.dataPaths[ class_ ])
                elif len(self.dataPaths[class_])<self.minLen:
                    self.minLen=len(self.dataPaths[class_])
                break
        if self.batchSize>self.minLen:
            self.batchSize=self.minLen;

        self.dataGen=self._getNextBatch()

    def resetGenerater(self):
        self.dataGen = self._getNextBatch()

    def getNextBatch(self):
        return next(self.dataGen)

    def getGenerater(self):
        return self._getNextBatch()

    def _getNextBatch(self):
        for i in range(self.minLen//self.batchSize):
            data = []
            for class_ in self.dataClasses:
                images = []
                for j in range(self.batchSize):
                    imgPath = os.path.join(self.rootPath,class_, self.dataPaths[class_][j])
                    img = cv2.imread(imgPath)
                    img=self._resizeAndPad(img)
                    images.append(img)
                data.append(images)
            yield data

    def _resizeAndPad(self, im):
        desired_size = self.imageSize
        old_size = im.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio)for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        return new_im