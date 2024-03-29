# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:39:33 2019

@author: priya

trainset_df dataframe is fed to svm classifier
"""
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

predictor_path = "C:\\Users\\priya\\Anaconda3\\shape_predictor_68_face_landmarks.dat"
faces_folder_path = "C:\\Users\\priya\\Anaconda3\\YALE\\faces"
   
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
Train=np.zeros((15 * split,68)) # Matrix of distances
z = 0
target = np.zeros((15 * split,1))
target_index = 0

rep_landmarks = list()
for subject in range(1, 16):
    
    subject_landmarks = []
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]

    for image_name in images_list[:split]:
        image_path = faces_folder_path + "\\" + image_name
        image = cv2.imread(image_path,-1)
        #print(image_name)
        target[target_index] = int(image_name[7:9])
        target_index += 1
        
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
   

Train = np.append(Train, target, axis=1)
print(Train.shape)

trainset_df = pd.DataFrame(data=Train, index=[x for x in range(120)], columns=[x for x in range(69)])
#print(trainset_df)

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(trainset_df[[x for x in range(68)]], trainset_df[68])

