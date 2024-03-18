#https://github.com/zonghua94/mnist
import pickle
import os
import glob
import pandas as pd
import numpy as np
#import cv2
from PIL import Image
import csv
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report 
###############################################################################

###############################################################################    
def relabel(labels):
    for idx,item in enumerate(labels):
        if item < 58 :
            labels[idx] = 0     #  number
        else:
            labels[idx] = 1     # character
    return labels
###############################################################################    
def f_score(p,r):
  f = (2*p*r)/(p+r)
  return f    

def main():
    
  #  root='/home/sawan/sawan_data/NIST/by_class/bin0/'
          infile = open('data_all_ml.pkl','rb')
          (X,Y) = pickle.load(infile)
          
          infile.close()
          '''
          heading = ['Size','Train','Test','LR_Acc','LR_Precision','LR_Recall','LR_F','NB_Acc','NB_Precision','NB_Recall','NB_F','SVM_Acc','SVM_Precision','SVM_Recall','SVM_F']
          file1 = open('results_2_class.csv','a')
          wtr = csv.writer(file1)
          wtr.writerow(heading)
          file1.close()
          '''
          Y=relabel(Y)
          #Y = to_categorical(Y, num_classes=2) 
          for i in range(10):
             
              A=np.array(X[:(i+1)*5500])
              B=np.array(Y[:(i+1)*5500])
             # A=X[:55000]
            #  B=Y[:55000]
              size=len(B)    
              actual_train_size=0.9*len(B)
              actual_test_size=0.1*len(B)
              
             
              #print(categorical_labels)
              X_train,X_test,Y_train,Y_test = train_test_split(A,B,test_size=0.1)
              print(len(X_test))
              print(len(Y_test))  
              #test_X = relabel(test_X)
             # Y_test = relabel(Y_test)
             # test_X = to_categorical(test_X, 2)
             # Y_test = to_categorical(Y_test, 2)
              from sklearn.linear_model import LogisticRegression
              model = LogisticRegression(solver='sag')
              
              from sklearn.naive_bayes import GaussianNB
              model1 = GaussianNB()
          
              
              from sklearn.svm import LinearSVC
              model2=LinearSVC()
              
              
              model.fit(X_train, Y_train)
              print(model)
              
              expected = Y_test
              
              predicted = pd.DataFrame(model.predict(X_test))
              accuracy = metrics.accuracy_score(np.round(predicted), Y_test)
              precision = metrics.precision_score(np.round(predicted), Y_test)
              recall = metrics.recall_score(np.round(predicted), Y_test)
              
              model1.fit(X_train, Y_train)
              print(model1)
            
              predicted1 = pd.DataFrame(model1.predict(X_test))
              accuracy1 = metrics.accuracy_score(np.round(predicted1), Y_test)
              precision1 = metrics.precision_score(np.round(predicted1), Y_test)
              recall1 = metrics.recall_score(np.round(predicted1), Y_test)
              
              model2.fit(X_train, Y_train)
              print(model2)
             
              predicted2 = pd.DataFrame(model2.predict(X_test))
              accuracy2 = metrics.accuracy_score(np.round(predicted2), Y_test)
              precision2 = metrics.precision_score(np.round(predicted2), Y_test)
              recall2 = metrics.recall_score(np.round(predicted2), Y_test)
              
        
              
              
              print("Logistic:")
              print(classification_report(expected, predicted))	
             # print(classification_report(expected, model2.predict(X_test)))	
              #report=classification_report(expected, predicted, output_dict=True)
              #df=pd.DataFrame(report).transpose()
             # print(expected[0],model2.predict(X_test[0].reshape(1,-1)))
              print("GaussianNB:") 	
              print(classification_report(expected, predicted1))		
              #report1=classification_report(expected, predicted1, output_dict=True)
         #     df1=pd.DataFrame(report1).transpose()
          
              
              print("SVM:") 	
              print(classification_report(expected, predicted2))		
          #    report2=classification_report(expected, predicted2, output_dict=True)
           #   df2=pd.DataFrame(report2).transpose()
              
        
              row = [size, actual_train_size, actual_test_size,accuracy, precision, recall, f_score(precision,recall), accuracy1, precision1, recall1, f_score(precision1,recall1),                         accuracy2      , precision2, recall2,f_score(precision2,recall2)]
        
              file1 = open('results_2_class.csv','a')
              wtr = csv.writer(file1)
              wtr.writerow(row)
              file1.close()
              
if __name__=='__main__':
    main()
#/home/sawan/sawan_data/NIST/by_class/??/train_??/train_??_*.png