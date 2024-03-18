import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import classification_report,precision_score,accuracy_score,recall_score
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

def relabel(labels):
    for idx,item in enumerate(labels):
        if item%2 ==0 :
            labels[idx] = 0     #  number
        else:
            labels[idx] = 1     # character
    return labels

def f_score(p,r):
  f = (2*p*r)/(p+r)
  return f    

def cnn_model():
 mod = Sequential()
 
 mod.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
 mod.add(Activation('relu'))
 BatchNormalization(axis=-1)
 mod.add(Conv2D(32, (3, 3)))
 mod.add(Activation('relu'))
 mod.add(MaxPooling2D(pool_size=(2,2)))
 
 BatchNormalization(axis=-1)
 mod.add(Conv2D(64,(3, 3)))
 mod.add(Activation('relu'))
 BatchNormalization(axis=-1)
 mod.add(Conv2D(64, (3, 3)))
 mod.add(Activation('relu'))
 mod.add(MaxPooling2D(pool_size=(2,2)))
 
 mod.add(Flatten())
 # Fully connected layer
 
 BatchNormalization()
 mod.add(Dense(512))
 mod.add(Activation('relu'))
 BatchNormalization()
 mod.add(Dropout(0.2))
 mod.add(Dense(2))
 mod.add(Activation('softmax'))
 # model.add(Convolution2D(10,3,3, border_mode='same'))
 # model.add(GlobalAveragePooling2D())
 
 
 mod.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
 
 return mod

model=cnn_model()
infile = open('data_all_dl.pkl','rb')
(X,Y) = pickle.load(infile)
infile.close()
Y=relabel(Y)
for i in range(10):
             
              A=np.array(X[:(i+1)*5500])
              B=np.array(Y[:(i+1)*5500]) 


              size=len(B)    
              actual_train_size=0.9*len(B)
              actual_test_size=0.1*len(B)
              '''
              heading = ['Size','Train','Test','CNN_Acc','CNN_Precision','CNN_Recall','CNN_F']
              file1 = open('results_cnn_2.csv','a')
              wtr = csv.writer(file1)
              wtr.writerow(heading)
              file1.close()
              ''' 
              #print(categorical_labels)
              X_train,X_test,Y_train,Y_test = train_test_split(A,B,test_size=0.1)
              
              
              X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
              X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
              
              X_train = X_train.astype('float32')
              X_test = X_test.astype('float32')
              
              X_train/=255
              X_test/=255
              
              Y_train = to_categorical(Y_train, num_classes = 2)
              Y_test = to_categorical(Y_test, num_classes = 2)
              
              gen = ImageDataGenerator()
              
              test_gen = ImageDataGenerator()
              train_generator = gen.flow(X_train, Y_train, batch_size=256)
              test_generator = test_gen.flow(X_test, Y_test, batch_size=256)
              
              model.fit_generator(train_generator, steps_per_epoch=size//256, epochs=1,validation_data=test_generator, validation_steps=size//2560)
              predicted2 = pd.DataFrame(model.predict(X_test))
              model.summary()
              accuracy2 = model.evaluate(X_test, Y_test)
              precision2 = metrics.precision_score(np.round(predicted2), Y_test,average='micro')
              recall2 = metrics.recall_score(np.round(predicted2), Y_test,average='micro')
             
              row = [size,actual_train_size, actual_test_size,accuracy2[1],precision2,recall2,f_score(precision2,recall2)]
              file1 = open('results_cnn_2.csv','a')
              wtr = csv.writer(file1)
              wtr.writerow(row)
              file1.close()