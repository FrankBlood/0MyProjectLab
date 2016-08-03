import os
import random
import re

filelist = os.listdir('/work/dingheng/WHUME/dataset/id_card/zh_chars_imgs/')
images = {}
name = {}
for files in filelist:
    imagefile = os.listdir('/work/dingheng/WHUME/dataset/id_card/zh_chars_imgs/'+files)
    fp = open('pic_test_imgs.txt', 'a')
    for image in imagefile:
        fp.write('/work/dingheng/WHUME/dataset/id_card/zh_chars_imgs/'+files+'/'+image+' '+files+' \n')
    fp.close()