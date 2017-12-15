import tensorflow as tf
import time
import threading
import numpy as np
import re
import glob
from PIL import Image

BUCKETS_DIR = "images/"
PROCESSED_IMAGES_DIR = "images/processed/"

def data_iterator(set='train',batch_size = 32):
    '''
    Python data generator to facilitate mini-batch training
    Arguments:
        set - 'train','valid','test' sets
        batch_size - integer (Usually 32,64,128, etc.)
    '''
    train_dict = np.load(BUCKETS_DIR + set+'_buckets.npy').tolist()
    print "Length of %s data: "%set,np.sum([len(train_dict[x]) for x in train_dict.keys()])
    for keys in train_dict.keys():
        train_list = train_dict[keys]
        N_FILES = (len(train_list)//batch_size)*batch_size
        for batch_idx in xrange(0,N_FILES,batch_size):
            train_sublist = train_list[batch_idx:batch_idx+batch_size]
            imgs = []
            batch_forms = []
            for x,y in train_sublist:
                imgs.append(np.asarray(Image.open(PROCESSED_IMAGES_DIR+x).convert('YCbCr'))[:,:,0][:,:,None])
                batch_forms.append(y)
            imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
            lens = [len(x) for x in batch_forms]

            mask = np.zeros((batch_size,max(lens)),dtype=np.int32)
            Y = np.zeros((batch_size,max(lens)),dtype=np.int32)
            for i,form in enumerate(batch_forms):
                mask[i,:len(form)] = 1
                Y[i,:len(form)] = form
            yield imgs, Y, mask
