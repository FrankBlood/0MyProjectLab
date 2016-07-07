import cv2
import numpy as np

class ImDetection(object):

    def __init__(self, dirname):
        self.dirname = dirname

    def getImage(self):
        print 'Succeful getImage!'
        return cv2.imread(self.dirname)

    def overlap(self, width, length, size = 1):
        overlapFile = []
        img = self.getImage()
        (imgWidth, imgLength, path) = np.shape(img)
        if width >= imgWidth or length >= imgLength:
            print 'Input Error'
            return
        rangeWidth = imgWidth - width + 1
        rangeLength = imgLength - length + 1
        for i in range(rangeWidth):
            for j in range(rangeLength):
                overlapImg = img[i:i+imgWidth, j:j+imgLength, :]
                overlapFile.append(overlapImg)
        print 'Sucessful overlap!'
        return overlapFile

    def window(self, width, length):
        windowFile = []
        img = self.getImage()
        (imgWidth, imgLength, path) = np.shape(img)
        if imgWidth / width < 1 or imgLength / length < 1:
            print 'Input Error'
            return
        rangeWidth = imgWidth / width
        rangeLength = imgLength / length
        for i in range(rangeWidth):
            for j in range(rangeLength):
                windowImg = img[i*imgWidth : (i+1) * imgWidth, j*imgWidth : (j+1)*imgLength, :]
                windowFile.append(windowImg)
        print 'Sucessful window!'

    def crop(self, width, length, amount):
        cropFile = []
        overlapFile = self.overlap(width, length)
        if amount > len(overlapFile):
            print 'Input Error'
            return
        rarray = np.random.randint(0, len(overlapFile), size=amount)
        for i in rarray:
            cropFile.append(overlapFile[i])
        print 'Sucessful crop!'
        return cropFile

if __name__ == '__main__':
    detection = ImDetection('classification.jpg')
    overlapFile = detection.overlap(362, 504)
    windowFile = detection.window(20, 40)
    cropFile = detection.crop(20, 40, 10)
    print np.shape(detection.getImage())
    print np.shape(overlapFile)
    print np.shape(windowFile)
    print np.shape(cropFile)
    print detection
    print detection.dirname