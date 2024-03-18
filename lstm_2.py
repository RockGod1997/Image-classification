import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,GRU
from keras.layers.normalization import  BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import pickle
import csv
from sklearn import metrics
from sklearn.metrics import classification_report,precision_score,accuracy_score,recall_score
def relabel(labels):
    for idx,item in enumerate(labels):
        if item%2 ==0 :
            labels[idx] = 0     #  number
        else:
            labels[idx] = 1     # character
    return labels
#################################################################################################

#################################################################################################
def lstm_Model():

    model = Sequential()



    model.add(LSTM(32, return_sequences=True, activation= 'relu', input_shape = (28,28) ) )

    model.add(LSTM(46, activation='relu', input_shape=(28, 28)))


    model.add(Dropout(0.2))
    
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model
#################################################################################################
def f_score(p,r):
  f = (2*p*r)/(p+r)
  return f   
   
def main():
    
    model2= lstm_Model()
    
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
                 
                   
                  #print(categorical_labels)
                  X_train,X_test,Y_train,Y_test = train_test_split(A,B,test_size=0.1)
                  
                  
                  
                  X_train = [x.reshape((-1, 28, 28)) for x in X_train]
                  X_train = np.array(X_train).reshape((-1, 28, 28))
                  
                  
                  X_test = [x.reshape((-1, 28, 28)) for x in X_test]
                  X_test = np.array(X_test).reshape((-1, 28, 28))
                  
                  Y_train = to_categorical(Y_train, num_classes = 2)
                  Y_test = to_categorical(Y_test, num_classes = 2)
                  
                  filepath = '/home/sawan/sawan_data/NIST/by_class/Weights2LS/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
                  
                  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
                  
                  callbacks_list = [checkpoint]
                  
                  batch_size = 256
                  epochs = 1
                  
                 
                  model2.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(X_test, Y_test))
                  predicted2 = pd.DataFrame(model2.predict(X_test))
                  model2.summary()
                  accuracy2 = model2.evaluate(X_test, Y_test)
                  precision2 = metrics.precision_score(np.round(predicted2), Y_test,average='micro')
                  recall2 = metrics.recall_score(np.round(predicted2), Y_test,average='micro')
                  
                  row = [size,actual_train_size, actual_test_size,accuracy2[1],precision2,recall2,f_score(precision2,recall2)]
                  with open('results_lstm_2.csv','a') as file1:
                    wtr = csv.writer(file1)
                    wtr.writerow(row)
                    file1.close()
if __name__=='__main__':
    main()