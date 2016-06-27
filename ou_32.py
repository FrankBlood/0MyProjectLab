import os
import random

filelist = os.listdir('./ImageCLEF2013TrainingSet')
images = {}
name = {}

print filelist

for files in filelist:
    if files.endswith('.txt'):
        continue
    if files == '.directory':
        continue
    if files == 'COMP':
        continue
    images[filelist.index(files)] = []
    name[filelist.index(files)] = files
    imagefile = os.listdir('./ImageCLEF2013TrainingSet/'+files)
    for image in imagefile:
        if image.endswith('.jpg'):
            images[filelist.index(files)].append('/'+files+'/'+image)

name_data = open('name_data_32.txt','w')
train_data = open('train_data_32.txt','w')

for (ID, imagefile) in images.items():
    try:
        for image in imagefile:
            train_data.write(str(image)+'\t'+str(ID)+'\n')
    except:
        print ID, images[ID]
train_data.close()

for (ID, files) in name.items():
    name_data.write(str(files)+'\t'+str(ID)+'\n')
name_data.close()
