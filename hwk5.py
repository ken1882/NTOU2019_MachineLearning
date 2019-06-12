import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from keras.optimizers import Adam
from keras import backend as K
import pickle

Flag_ShowSample = False

fruit_dir = './fruits-360'

class_id = {}
class_name={}
training_set_size = 0
test_set_size = 0

MaxTrainNum = int(53177 // 4)
MaxTestNum = int(17845 // 4)

try:
  with open('header.pkdat', 'rb') as file:
    meta = pickle.load(file)
    training_set_size = meta['train_size']
    test_set_size = meta['test_size']
    class_id = meta['clsid']
    class_name = meta['clsname']
  print("Header loaded")
except Exception:
  for idx,x in enumerate(glob.iglob(os.path.join(fruit_dir,os.path.join('Training','*')))):
    # if idx > MaxTrainNum:
    #   break
    training_set_size += len(glob.glob(os.path.join(x,'*.jpg')))
    class_id[x.split('\\')[-1]] = idx
    class_name[idx] = x.split('\\')[-1]

  for idx,x in enumerate(glob.iglob(os.path.join(fruit_dir,os.path.join('Test','*')))):
    # if idx > MaxTestNum:
    #   break
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
try:
  with open('train.pkdat', 'rb') as file:
    meta = pickle.load(file)
    x_train = meta['train']
    training_label = meta['label']
    training_id = meta['id']
  print("Training data loaded")
except Exception:
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
  with open('train.pkdat', 'wb') as file:
    pickle.dump({'train':x_train, 'label': training_label, 'id':training_id}, file)

print("Processing testing cases")
test_id = 0
try:
  with open('test.pkdat', 'rb') as file:
    meta = pickle.load(file)
    x_test = meta['train']
    test_label = meta['label']
    test_id = meta['id']
  print("Test data loaded")
except Exception:
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
  with open('test.pkdat', 'wb') as file:
    pickle.dump({'train':x_test, 'label': test_label, 'id':test_id}, file)

def show_sample():
  plt.figure(figsize=(20,8))
  for splt,idx in enumerate(np.random.permutation(x_train.shape[0])[:5]):
    plt.subplot(2,5,splt+1)
    img = x_train[idx,:,:,:3]
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    plt.imshow(img[:,:,::-1])
    plt.title('training:{},id#:{}'.format(class_name[training_label[idx]],idx))
    plt.axis('Off')

  for splt,idx in enumerate(np.random.permutation(x_test.shape[0])[:5]):
    plt.subplot(2,5,splt+6)
    img = x_train[idx,:,:,:3]
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    plt.imshow(img[:,:,::-1])
    plt.title('test:{},id#:{}'.format(class_name[test_label[idx]], idx))
    plt.axis('Off')
  plt.show()


# Prepare Training/Validation set 
epochs   = 50
batchsize= 60

x_train  = x_train / 255
x_test   = x_test / 255

print("Spliting training samples")
x_train0, x_val, train0_label, val_label = train_test_split(x_train, training_label, test_size=0.1, random_state=303, shuffle=False)

if Flag_ShowSample:
  show_sample()
  x_train = None
else:
  x_train = None

# one-hot encoding category tags
y_train0  = to_categorical(train0_label)
y_val     = to_categorical(val_label)
y_test   = to_categorical(test_label)

# Data Augmentation
datagen = ImageDataGenerator(vertical_flip=True,horizontal_flip=True)

def define_network_architecture_5x5():
  model = Sequential()
  model.add(Conv2D(16,(5,5),strides=(1,1),input_shape=(100,100,4),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Conv2D(32,(5,5),strides=(1,1),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64,(5,5),strides=(1,1),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(128,(5,5),strides=(1,1),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dense(1024,activation='relu'))
  model.add(Dropout(0.8))
  model.add(Dense(256,activation='relu'))
  model.add(Dropout(0.8))
  model.add(Dense(103,activation='softmax'))
  model.summary()
  print('the total number of layers:{}'.format(len(model.layers)))    
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  return model

model_5x5 = define_network_architecture_5x5()

def train_model_5x5():
  print("Start training 5x5")
  s = time.time()
  history_5x5 = model_5x5.fit_generator(datagen.flow(x_train0, y_train0, batch_size=batchsize),steps_per_epoch=x_train0.shape[0]//batchsize, epochs=epochs,
    validation_data=datagen.flow(x_val, y_val))
  print('5x5 training time:{}'.format(time.time()-s))
  plt.figure(figsize=(8,4))
  plt.title('model 5x5')
  plt.plot(history_5x5.history['acc'],label='training')
  plt.plot(history_5x5.history['val_acc'],label='validation')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.grid('On')
  plt.legend()
  plt.show()


def define_network_architecture_3x3():
  data = Input(shape=(100,100,4))
  x1   = Conv2D(8,(3,3),strides=(1,1),padding='same',activation='relu')(data)
  x2   = Conv2D(8,(3,3),strides=(1,1),padding='same',activation='relu')(x1)
  z    = Concatenate()([x1,x2])
  z    = MaxPooling2D(pool_size=(2,2))(z)
  
  x3   = Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu')(data)
  x4   = Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu')(x3)
  z    = Concatenate()([x3,x4])
  z    = MaxPooling2D(pool_size=(2,2))(z)
  
  x5   = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu')(data)
  x6   = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu')(x5)
  z    = Concatenate()([x5,x6])
  z    = MaxPooling2D(pool_size=(2,2))(z)

  x7   = Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu')(data)
  x8   = Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu')(x7)
  z    = Concatenate()([x7,x8])
  z    = MaxPooling2D(pool_size=(2,2))(z)

  z    = Flatten()(z)
  z    = Dense(1024,activation='relu')(z)
  z    = Dropout(0.8)(z)
  z    = Dense(256,activation='relu')(z)
  z    = Dropout(0.8)(z)
  z    = Dense(103,activation='softmax')(z)
  model = Model(inputs=data,outputs=z)
  model.summary()
  print('the total number of layers:{}'.format(len(model.layers)))
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  return model

model_3x3 = define_network_architecture_3x3()

def train_model_3x3():
  print("Start training 3x3")
  s = time.time()
  history_3x3 = model_3x3.fit_generator(datagen.flow(x_train0, y_train0, batch_size=batchsize),steps_per_epoch=x_train0.shape[0]//batchsize, epochs=epochs,
    validation_data=datagen.flow(x_val, y_val))

  print('3x3 training time:{}'.format(time.time()-s))
  plt.figure(figsize=(8,4))
  plt.title('model 3x3')
  plt.plot(history_3x3.history['acc'],label='training')
  plt.plot(history_3x3.history['val_acc'],label='validation')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.grid('On')
  plt.legend()
  plt.show()

scores_5x5 = model_5x5.evaluate(x_test,y_test)
scores_3x3 = model_3x3.evaluate(x_test,y_test)
print('accuracy 5x5:{:.4f}, 3x3:{:.4f}'.format(scores_5x5[1],scores_3x3[1]))