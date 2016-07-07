import numpy as np
import cv2
class Detection(object):

	def __init__(self, dirname):
		self.dirname = dirname

    # def getImage(self):
    #     img = cv2.imread(self.dirname)
    #     cv2.imshow('name', img)

	def window(self):
		print 'Sucessful!'

	def overlap(self):
		pass

	def crop(self):
		pass

if __name__ == "__main__":
    detection = Detection('classification.jpg')
    # detection.getImage()
    detection.window()
    print detection.dirname
    print detection