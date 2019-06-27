# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:39:33 2019

@author: priya

svm classifier from sklearn
"""
'''
    Accuracy         Image Order
93.3333333333 -> [4, 1, 3, 0, 2, 5, 9, 8, 7, 10, 6]
60.0 -> [3, 4, 5, 1, 2, 6, 7, 8, 10, 0, 9]
93.33333333333333 -> [0, 3, 7, 1, 10, 9, 6, 4, 8, 2, 5]
95.5555555556 -> [2, 0, 8, 7, 10, 4, 9, 1, 5, 6, 3]
62.2222222222 -> [8, 0, 4, 1, 7, 10, 2, 5, 9, 6, 3]
66.6666666667 -> [4, 7, 1, 5, 10, 0, 8, 2, 3, 6, 9]
91.1111111111 -> [0, 9, 6, 8, 1, 5, 4, 10, 3, 7, 2]
68.8888888889 -> [3, 10, 4, 8, 0, 7, 5, 2, 9, 1, 6]
'''

import pandas as pd
import os
import dlib
import cv2
from imutils import face_utils
import re
from collections import OrderedDict
import numpy as np
import math
from sklearn import svm
import random

def atoi(text):
    if text.isdigit():
        return int(text) 
    else:
        return text

def natural_keys(text):
    l=[]
    for c in re.split('(\d+)', text):
       l.append(atoi(c))     
    return(l)    

def euclidean_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "faces"
   
file_list=os.listdir(faces_folder_path)
file_list = [x for x in file_list if re.search("^.*pgm$", x)]
file_list.sort(key=natural_keys)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

split = 8
Train = np.zeros((15 * split,68)) 
Test = np.zeros((15 * (11-split),68)) # Matrix of distances

train = 0
test = 0
train_target = np.zeros((15 * split,1))
test_target = np.zeros((15 * (11-split),1))
target_index = 0
test_target_index = 0

#indices = list(range(11))
#random.shuffle(indices)
indices = [5, 0, 8, 7, 10, 4, 9, 1, 2, 6, 3]

rep_landmarks = list()
for subject in range(1, 16):
    
    subject_landmarks = []
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]
    images_list = [images_list[i] for i in indices]

    for image_name in images_list:
        image_path = faces_folder_path + "\\" + image_name
        image = cv2.imread(image_path,-1)

        faces = detector(image, 1)
        for i, face in enumerate(faces):
            shape = predictor(image, face)
            shape = face_utils.shape_to_np(shape)
                
        representative_landmark = [ sum([point[0] for point in shape]) / 68, sum([point[1] for point in shape]) / 68]
        distances = [0] * 68
        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            index = i
            for (x, y) in shape[i:j]:
                distances[index] = euclidean_distance(representative_landmark, (x,y))
                index += 1
        if images_list.index(image_name) < split:
            Train[train,:] = distances
            train_target[target_index] = int(image_name[7:9])
            target_index += 1
            train += 1
        else:
            Test[test,:] = distances
            test_target[test_target_index] = int(image_name[7:9])
            test_target_index += 1
            test += 1
            
   
Train = np.append(Train, train_target, axis=1)

trainset_df = pd.DataFrame(data=Train, index=[x for x in range(120)], columns=[x for x in range(69)])
X_train = trainset_df.iloc[:,list(range(0,68))]
y_train = trainset_df.iloc[:, 68]

clf = svm.SVC(kernel = "poly", C = 1.0)
clf.fit(X_train, y_train)

testset_df = pd.DataFrame(data=Test, index=[x for x in range(45)], columns=[x for x in range(68)])
y_pred = clf.predict(testset_df)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_target, y_pred) * 100, "->", indices)
