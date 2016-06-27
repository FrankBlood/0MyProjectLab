import os
import random

filelist = os.listdir('./ImageCLEFTestSetGROUNDTRUTH')
images = {}
name = {}

images[0] = []
images[1] = []
print filelist

for files in filelist:
    if files.endswith('.txt'):
        continue
    if files == '.directory':
        continue
    
    if files == 'COMP':
        name[0] = 'COMP'
        imagefile = os.listdir('./ImageCLEFTestSetGROUNDTRUTH/'+files)
        for image in imagefile:
            if image.endswith('.jpg'):
                images[0].append('/'+files+'/'+image)
    else:
        name[1] = 'OTHERS'
        imagefile = os.listdir('./ImageCLEFTestSetGROUNDTRUTH/'+files)
        for image in imagefile:
            if image.endswith('.jpg'):
                images[1].append('/'+files+'/'+image)
name_data = open('test_name_data_2.txt','w')
train_data = open('test_data_2.txt','w')

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

