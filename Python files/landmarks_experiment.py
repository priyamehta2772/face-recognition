# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:00:00 2019

@author: LENOVO
"""
'''
    Accuracy            Train images combination
28.88888888888889 -> [4, 1, 3, 0, 2, 5, 9, 8, 7, 10, 6]
28.88888888888889 -> [3, 4, 5, 1, 2, 6, 7, 8, 10, 0, 9]
28.88888888888889 -> [0, 3, 7, 1, 10, 9, 6, 4, 8, 2, 5]
33.333333333333336  -> [2, 0, 8, 7, 10, 4, 9, 1, 5, 6, 3]
20.0 -> [8, 0, 4, 1, 7, 10, 2, 5, 9, 6, 3]
22.22222222222222 -> [4, 7, 1, 5, 10, 0, 8, 2, 3, 6, 9]
26.666666666666668 -> [0, 9, 6, 8, 1, 5, 4, 10, 3, 7, 2]
24.444444444444443 -> [3, 10, 4, 8, 0, 7, 5, 2, 9, 1, 6]
'''
import random
import os
import dlib
import cv2
from imutils import face_utils
import re
import numpy as np
import math

def euclidean_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

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

def get_subject_name(subject):
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
    return subject

def get_subject_images(subject, file_list):
    return [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]

predictor_path = "C:\Anaconda3\shape_predictor_68_face_landmarks.dat"
dataset_path = "E:\Talentsprint_WE\FaceExpressionRecognitionUsingCNN\\faces"

file_list = os.listdir(dataset_path)
file_list = [x for x in file_list if re.search("^.*pgm$", x)]
file_list.sort(key = natural_keys)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#indices = list(range(11))
#random.shuffle(indices)
indices = [2, 0, 8, 7, 10, 4, 9, 1, 5, 6, 3]

split = 8
no_of_classes = 15
images_per_subject = 11
no_of_landmarks =  68

Train = np.zeros((no_of_classes * split, 2))
Test = np.zeros((no_of_classes * (images_per_subject - split), 2))
train = 0
test = 0
for subject in range(1, no_of_classes + 1):
    
    subject = get_subject_name(subject)
    subject_landmarks = []
    images_list = get_subject_images(subject, file_list)
    images_list = [images_list[i] for i in indices]
    for image_name in images_list:
    
        image_path = dataset_path + "\\" + image_name
        image = cv2.imread(image_path,-1)
        faces_in_image = detector(image, 1)
        
        for i, face in enumerate(faces_in_image):
            shape = predictor(image, face)
            shape = face_utils.shape_to_np(shape)
                
        representative_landmark = [ sum([point[0] for point in shape]) / 68, sum([point[1] for point in shape]) / 68]
        
        if images_list.index(image_name) < split:
            Train[train,:] = representative_landmark
            train += 1
        else:
            Test[test,:] = representative_landmark
            test += 1

count = 0
img_count = 0
for test_landmark in Test:
    img_count += 1
    distances = []
    for subject_landmark in Train:
        dist = euclidean_distance(subject_landmark, test_landmark)
        distances.append(dist)
    
    if int(img_count//3) == int(distances.index(min(distances)) // split):
        count +=  1
        
print(count*100/img_count, "->", indices) 

    
    