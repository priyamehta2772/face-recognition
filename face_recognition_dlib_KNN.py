# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:52:04 2019

@author: LENOVO
"""
"""
    Accuracy               Train images combination
88.88888888888889 -> [4, 1, 3, 0, 2, 5, 9, 8, 7, 10, 6]
57.77777777777778 -> [3, 4, 5, 1, 2, 6, 7, 8, 10, 0, 9]
93.33333333333333 -> [0, 3, 7, 1, 10, 9, 6, 4, 8, 2, 5]
93.33333333333333 -> [2, 0, 8, 7, 10, 4, 9, 1, 5, 6, 3]
55.55555555555556 -> [8, 0, 4, 1, 7, 10, 2, 5, 9, 6, 3]
62.22222222222222 -> [4, 7, 1, 5, 10, 0, 8, 2, 3, 6, 9]
91.11111111111111 -> [0, 9, 6, 8, 1, 5, 4, 10, 3, 7, 2]
62.22222222222222 -> [3, 10, 4, 8, 0, 7, 5, 2, 9, 1, 6]
"""
import os
import dlib
import cv2
from imutils import face_utils
import re
from collections import OrderedDict
import numpy as np
import math
import random
    
def filter_pgm(dataset_path):
    file_list = os.listdir(dataset_path)
    file_list = [x for x in file_list if re.search("^.*pgm$", x)]
    file_list.sort(key=natural_keys)
    return file_list

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

dataset_path = "faces"
file_list = filter_pgm(dataset_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
no_of_classes = 15
images_per_class = 11
no_of_landmarks =  68

Train = np.zeros((no_of_classes * split, no_of_landmarks)) # Matrix of distances
Test = np.zeros((no_of_classes * (images_per_class - split), no_of_landmarks))
train = 0
test = 0
rep_landmarks = list()
#indices = list(range(11))
#random.shuffle(indices)

indices = [0, 3, 7, 1, 10, 9, 6, 4, 8, 2, 5]
for subject in range(1, no_of_classes + 1):
    
    subject_landmarks = []
    if len(str(subject)) == 1:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]
    images_list = [images_list[i] for i in indices]

    for image_name in images_list:
        image_path = dataset_path + "\\" + image_name
        image = cv2.imread(image_path,-1)

        faces = detector(image, 1)
        for i, face in enumerate(faces):
            shape = predictor(image, face)
            shape = face_utils.shape_to_np(shape)
                
        representative_landmark = [ sum([point[0] for point in shape]) / no_of_landmarks, sum([point[1] for point in shape]) / no_of_landmarks]
        distances = [0] * no_of_landmarks
        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            index = i
            for (x, y) in shape[i:j]:
                distances[index] = euclidean_distance(representative_landmark, (x,y))
                index += 1
        if images_list.index(image_name) < split:
            Train[train,:] = distances
            train += 1
        else:
            Test[test,:] = distances
            test += 1

count = 0
img_count = 0

for test_image_distances in Test:
    img_count += 1
    Train_diff = np.zeros((no_of_classes * split, no_of_landmarks))
        
    for index in range(len(Train)):
        Train_diff[index,:] = list(map(abs, np.subtract(Train[index], np.asarray(test_image_distances))))
    
    Train_diff_means = [0] * no_of_classes * split
    for index in range(len(Train_diff)):
        Train_diff_means[index] = np.mean(Train_diff[index])
    
    if int(Train_diff_means.index(min(Train_diff_means)) // split) == int((img_count - 1) // (images_per_class - split)):
        count += 1
            
print(count*100/img_count, "->", indices)


