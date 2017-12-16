'''
Based on code from - https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import glob
import pickle
import csv
from itertools import izip_longest

def image_to_feature_vector(image, size=(28, 28)):
  return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(2,2)):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("gray", gray)
  hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

  if imutils.is_cv2():
    hist = cv2.normalize(hist)
  else:
    cv2.normalize(hist, hist)
  return hist.flatten()

print("[INFO] describing images...")
images = []
neighbours = 52

error = 0
count = 0
counter = 0

f = open('model', 'rb')
model = pickle.load(f)
print "model loaded"
with open('image_path_label.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        if row == ['Path,Label']:
            continue
        label = row[1]
        direc_name = row[0][:len(row[0])-4]  
        images = []
        for name in glob.glob(direc_name+'/*'):
            images.append(name)
         

        features = []
        outputs = []
        for (i, imagePath) in enumerate(images):
          image = cv2.imread(imagePath)
          pixels = image_to_feature_vector(image)
          hist = extract_color_histogram(image)
         
          features.append(hist)
          if i > 0 and i % 1000 == 0:
            print("processed {}/{}".format(i, len(images)))

        
        features = np.array(features)
        outputs = model.predict(features)
        outputs = outputs.tolist()
        sentence = "".join(outputs)
        label = label.split(" ")
        label = "".join(label)
        count += len(label)
        
        if len(sentence) > len(label):
            sentence = sentence[:len(label)]

        for x,y in izip_longest(sentence, label, fillvalue='|'):
            if x != y:
                error += 1
        
        counter += 1
        if counter % 10 == 0:
            print ("At example", counter, "Error", (count-error)*100/count)

acc =  (count-error)*100/count
print "percent accuracy is ", acc

