'''
Based on code from  - https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
'''

# import the necessary packages
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
labels = []
neighbours = 52

for name in glob.glob('data/by_class/*'):
    sub_dir  = "data/by_class/" + name[14:] + "/train_" + name[14:] + "/*"
    for image_path in glob.glob(sub_dir):
        images.append(image_path)
        labels.append(image_path[14:16].decode("hex"))

features = []

for (i, imagePath) in enumerate(images):
  image = cv2.imread(imagePath)
  pixels = image_to_feature_vector(image)
  hist = extract_color_histogram(image)
 
  features.append(hist)
  if i > 0 and i % 1000 == 0:
    print("processed {}/{}".format(i, len(images)))

features = np.array(features)
labels = np.array(labels)
print("features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

print("evaluating accuracy...")

model = KNeighborsClassifier(n_neighbors=neighbours)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
pickle.dump(model,open( "model", "wb" ))


