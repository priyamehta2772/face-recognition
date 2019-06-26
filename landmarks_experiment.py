# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:00:00 2019

@author: LENOVO
"""
'''
    Accuracy            Train images combination
28.88888888888889 -> [2, 3, 1, 9, 5, 6, 0, 8, 7, 10, 4]
35.55555555555556 -> [3, 4, 5, 10, 6, 8, 9, 7, 2, 1, 0]
26.666666666666668 -> [8, 9, 1, 0, 10, 6, 5, 4, 7, 3, 2]
42.22222222222222 -> [9, 2, 3, 10, 5, 6, 0, 7, 8, 4, 1]
40.0 -> [10, 0, 5, 3, 6, 8, 9, 2, 1, 4, 7]
26.666666666666668 -> [5, 2, 10, 1, 7, 6, 4, 3, 0, 9, 8]
31.11111111111111 -> [5, 8, 1, 0, 3, 6, 2, 10, 9, 4, 7]
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
indices = [9, 2, 3, 10, 5, 6, 0, 7, 8, 4, 1]

split = 8
Train = np.zeros((15 * split, 2))
Test = np.zeros((15 * (11 - split), 2))
train = 0
test = 0
for subject in range(1, 16):
    
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

    
    