import os

from shutil import copyfile
from random import randint,shuffle
from PIL import Image
###############################################################################

def dir_check(location):
    
    if not os.path.exists(location):
        os.makedirs(location)

###############################################################################
def main():
    
    root='/home/sawan/sawan_data/NIST/by_class/'
    dst='dataset/'

    dir_check(dst+'train/')
    dir_check(dst+'val/')
    
    
    classes=os.listdir(root)
    
    for clas in classes:
        
        imlist=os.listdir(root+clas)
        shuffle(imlist)
        
        train=imlist[:2000]
        val=imlist[2001:4001]
        
        ## write train and val files
        
        dir_check(dst+'train/'+clas)
        dir_check(dst+'val/'+clas)
        
        for tmp in train:
            
            fname=str(randint(0,999999999999))
            img=Image.open(root+clas+'/'+tmp)
            Image.save(dst+'train/'+clas+'/'+fname+'.jpg',img)
        
        for tmp in val:
            
            fname=str(randint(0,999999999999))
            img=Image.open(root+clas+'/'+tmp)
            Image.save(dst+'val/'+clas+'/'+fname+'.jpg',img)
    

if __name__=='__main__':
    main()
