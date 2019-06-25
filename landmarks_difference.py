# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:52:04 2019

@author: LENOVO
"""

import os
import dlib
import cv2
from imutils import face_utils
import re
from collections import OrderedDict
import numpy as np
import math

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

predictor_path = "C:\Anaconda3\shape_predictor_68_face_landmarks.dat"
faces_folder_path = "E:\Talentsprint_WE\FaceExpressionRecognitionUsingCNN\\faces"
    #new_folder_path = "E:\Talentsprint_WE\detected_faces\\"
    
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

Train=np.zeros((120,68)) # Matrix of distances
z = 0

rep_landmarks = list()
for subject in range(1, 16):
    
    subject_landmarks = []
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]

    for image_name in images_list[:8]:
        image_path = faces_folder_path + "\\" + image_name
        image = cv2.imread(image_path,-1)
        #print(image_path)
        faces = detector(image, 1)
        for i, face in enumerate(faces):
            shape = predictor(image, face)
            shape = face_utils.shape_to_np(shape)
                
        representative_landmark = [ sum([point[0] for point in shape]) / 68, sum([point[1] for point in shape]) / 68]
        train_distances = [0] * 68
        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            index = i
            for (x, y) in shape[i:j]:
                train_distances[index] = euclidean_distance(representative_landmark, (x,y))
                index += 1
        Train[z,:] = train_distances
        z += 1


test_image = "E:\Talentsprint_WE\FaceExpressionRecognitionUsingCNN\\faces\subject11.wink.pgm"

test_image = cv2.imread(test_image,-1)
test_image_faces = detector(test_image, 1)
for i, face in enumerate(test_image_faces):
    test_shape = predictor(test_image, face)
    test_shape = face_utils.shape_to_np(test_shape)
        
test_landmarks = [ sum([point[0] for point in test_shape]) / 68, sum([point[1] for point in test_shape]) / 68]
test_distances = [0] * 68
for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    index = i
    for (x, y) in test_shape[i:j]:
        test_distances[index] = euclidean_distance(test_landmarks, (x,y))
        index += 1

Train_diff = np.zeros((120,68)) # Matrix of differences of distances

for index in range(len(Train)):
    Train_diff[index,:] = list(map(abs, np.subtract(Train[index], np.asarray(test_distances))))
    
Train_diff_means = [0] * 120
for index in range(len(Train_diff)):
    Train_diff_means[index] = np.mean(Train_diff[index])
    
print(Train_diff_means.index(min(Train_diff_means)))

###############################################################################
for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    
    #clone = image.copy()
    #cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    index = i
    for (x, y) in shape[i:j]:
        train_distances[index] = euclidean_distance(representative_landmark, (x,y))
        index += 1
        #cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=350, inter=cv2.INTER_CUBIC)
 
		# show the particular face part
    cv2.imshow("ROI", roi)
    cv2.imshow("Clone",clone)
    cv2.waitKey(0)
