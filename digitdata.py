#https://github.com/kmdanielduan/Logistic-Regression-on-MNIST-with-NumPy-from-Scratch
import os
import glob
import pickle
import numpy as np
#import cv2
from PIL import Image,ImageFilter
import sys

from keras.utils.np_utils import to_categorical

###############################################################################
def relabel(labels):
    for idx, item in enumerate(labels):
        if item <58 :
            labels[idx] = 0     #  number
        else:
            labels[idx] = 1     # character 
    return labels
#####################

###############################################################################
def main():
    
  
    X=[]
    Y=[]
    files=glob.glob('/home/sawan/sawan_data/NIST/by_class/??/train_??/train_??_0000*.png') #train_??_00[0-8]*.png
    #print(files)
    for file in files :
        
            
          im = Image.open(file).convert(mode='L')
          width = float(im.size[0])
          height = float(im.size[1])
          newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
          if width > height:  # check which dimension is bigger
              # Width is bigger. Width becomes 20 pixels.
              nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
              if (nheight == 0):  # rare case but minimum is 1 pixel
                  nheight = 1
                  # resize and sharpen
              img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
              wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
              newImage.paste(img, (4, wtop))  # paste resized image on white canvas
          else:
              # Height is bigger. Heigth becomes 20 pixels.
              nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
              if (nwidth == 0):  # rare case but minimum is 1 pixel
                  nwidth = 1
                  # resize and sharpen
              img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
              wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
              newImage.paste(img, (wleft, 4))      
         
          
        
          #X.append(flat_arr)
         # X.append(newImage)        
              arr=np.array(newImage)
              flat_arr = arr.ravel()
          X.append(arr)
          Y=np.append(Y,int(file[55:57],16))


          #categorical_labels = to_categorical(Y,2)
    
   
   # Y=relabel(Y)
    #print(arr)    
    #print(Z)
   # print(type(Y))
  #  print(len(flat_arr))
    #print(len(Y)) 
    #print(categorical_labels.shape[0])
    from sklearn.utils import shuffle
    X=np.asarray(X)
    Y=np.asarray(Y)
    (X, Y) = shuffle(X, Y, random_state=0)
    
    filename = 'data_ml2323.pkl'
    outfile = open(filename,'wb')
    pickle.dump((X,Y),outfile)
    outfile.close()          
if __name__=='__main__':
    main()
