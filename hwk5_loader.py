import numpy as np 
import pickle
import glob
import cv2
import os 

fruit_dir = './fruits-360'

class_id = {}
class_name={}
training_set_size = 0
test_set_size = 0

for idx,x in enumerate(glob.iglob(os.path.join(fruit_dir,os.path.join('Training','*')))):
  training_set_size += len(glob.glob(os.path.join(x,'*.jpg')))
  class_id[x.split('\\')[-1]] = idx
  class_name[idx] = x.split('\\')[-1]

for idx,x in enumerate(glob.iglob(os.path.join(fruit_dir,os.path.join('Test','*')))):
  test_set_size += len(glob.glob(os.path.join(x,'*.jpg')))

with open('header.pkdat', 'wb') as file:
  pickle.dump({
    'train_size': training_set_size,
    'test_size': test_set_size,
    'clsid': class_id,
    'clsname': class_name
  }, file)

print('total number of training images: {}, total number of test images: {}, total number of classes: {}'.format(training_set_size,test_set_size,len(class_id)))

label_dtype = np.uint8
x_train = np.zeros((training_set_size,100,100,4),dtype=np.uint8)
training_label = np.zeros((training_set_size,),dtype=label_dtype)
x_test = np.zeros((test_set_size,100,100,4),dtype=np.uint8)
test_label = np.zeros((test_set_size,),dtype=label_dtype)

print("Processing training cases")
training_id = 0

for x in glob.iglob(os.path.join(fruit_dir,os.path.join('Training','*'))):
  cid = class_id[x.split('\\')[-1]]
  for f in glob.glob(os.path.join(x,'*.jpg')):
    img = cv2.imread(f,cv2.IMREAD_COLOR)
    g   = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    x_train[training_id,:,:,:3] = img 
    x_train[training_id,:,:,3]  = g
    training_label[training_id] = cid
    training_id = training_id + 1
    
x_train = (x_train / 255).astype(np.float32)
with open('train.pkdat', 'wb') as file:
  pickle.dump({'train':x_train, 'label': training_label, 'id':training_id}, file)

x_train, training_label, training_id = None, None, None

print("Processing testing cases")
test_id = 0
for x in glob.iglob(os.path.join(fruit_dir,os.path.join('Test','*'))):
  cid = class_id[x.split('\\')[-1]]
  for f in glob.glob(os.path.join(x,'*.jpg')):
    img = cv2.imread(f,cv2.IMREAD_COLOR)
    g   = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    x_test[test_id,:,:,:3] = img
    x_test[test_id,:,:,3]  = g
    test_label[test_id] = cid
    test_id = test_id + 1

x_test = (x_test / 255).astype(np.float32)
with open('test.pkdat', 'wb') as file:
  pickle.dump({'train':x_test, 'label': test_label, 'id':test_id}, file)