# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:00:00 2019

@author: LENOVO
"""


import os
import dlib
import cv2
from imutils import face_utils
import re
predictor_path = "C:\Anaconda3\shape_predictor_68_face_landmarks.dat"
faces_folder_path = "E:\Talentsprint_WE\FaceExpressionRecognitionUsingCNN\\faces"
new_folder_path = "E:\Talentsprint_WE\detected_faces\\"

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

file_list=os.listdir(faces_folder_path)
file_list = [x for x in file_list if re.search("^.*pgm$", x)]
file_list.sort(key=natural_keys)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

rep_landmarks = list()
for subject in range(1, 16):
    
    subject_landmarks = []
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]
    subject_path = new_folder_path + str(subject) + "\\"
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)

    for image_name in images_list[:8]:
    
        image_path = faces_folder_path + "\\" + image_name
        image = cv2.imread(image_path,-1)
        #print(image_path)
        faces = detector(image, 1)
        for i, face in enumerate(faces):
            shape = predictor(image, face)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                
        representative_landmark = [ sum([point[0] for point in shape]) / 68, sum([point[1] for point in shape]) / 68]
        subject_landmarks.append(representative_landmark)
        #cv2.imwrite(subject_path + "\\" + image_name, image)
        #cv2.imshow(image_name, image)
        #cv2.waitKey(0)
    rep_landmarks.append(subject_landmarks)

test_image = "E:\Talentsprint_WE\FaceExpressionRecognitionUsingCNN\faces\subject01.wink.pgm"
faces = detector(image, 1)
for i, face in enumerate(faces):
    test_shape = predictor(image, face)
    (x, y, w, h) = face_utils.rect_to_bb(face)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    test_shape = face_utils.shape_to_np(test_shape)
    for (x, y) in test_shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
test_landmarks = [ sum([point[0] for point in test_shape]) / 68, sum([point[1] for point in test_shape]) / 68]

import math
def euclidean_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

distances = []
for person in rep_landmarks:
    print("Person:",person)
    train_test_distances = []
    for train_images_landmarks in person:
        print("TIL:",train_images_landmarks)
        dist = euclidean_distance(train_images_landmarks, test_landmarks)
        train_test_distances.append(dist)
    print("TTD:",train_test_distances)
    distances.append([rep_landmarks.index(person), min(train_test_distances)])
    
print("Class:", distances)
        
 '''       
import numpy as np  
image_batch = np.expand_dims(image, axis=0)
import keras
from keras.applications import resnet50
resnet_model = resnet50.ResNet50(weights = 'imagenet')'''
    
    
    