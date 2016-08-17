import numpy as np
import re
import sys

imgs_vector = {}
key = 0
with open(sys.argv[1], 'r') as vec:
    for line in vec.readlines():
        line = line.strip().split()
        img_name = line[0]
        img_name = re.split('/', img_name)[3]
        img_vec = line[1:]
        img_vec = map(float, img_vec)
        img_vec = np.array(img_vec)
        imgs_vector[img_name] = img_vec
        if key == 0:
            vec_len = len(img_vec)

with open('user.txt', 'r') as users:
    for imgs in users.readlines():
        imgs = imgs.strip().split()
        user_name = imgs[0]
        imgs = imgs[1:]
        imgs_num = len(imgs)
        user_vec = np.zeros(vec_len)
        for img in imgs:
            try:
                vector = imgs_vector[img]
                user_vec += vector
            except:
                continue
        user_vec /= imgs_num
        user_f = 1 + np.dot(user_vec, np.log2(user_vec)) / np.log2(vec_len)
        print user_name,
        print user_f,
        for i in user_vec:
            print i,
        print