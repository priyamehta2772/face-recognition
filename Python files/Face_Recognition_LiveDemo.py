# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:39:24 2019

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

def recognizer(Train, test_image_distances ):
    Train_diff = np.zeros((178, 68))
        
    for index in range(len(Train)):
        Train_diff[index,:] = list(map(abs, np.subtract(Train[index], np.asarray(test_image_distances))))
    
    Train_diff_means = [0] * 178
    for index in range(len(Train_diff)):
        Train_diff_means[index] = np.mean(Train_diff[index])
    
    return int(Train_diff_means.index(min(Train_diff_means)) // 11) + 1
            
no_of_classes = 15
images_per_class = 11
no_of_landmarks =  68

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

### Get the new face data
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_name = input('\n Enter your name: ')
no_of_classes += 1
face_id = no_of_classes
new_labels = {face_id:face_name}

print("\n Initializing face capture. Look the camera and wait ...")
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img, 1)
    for i, face in enumerate(faces):
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("E:\Talentsprint_WE\\face-recognition\\faces\\subject" + str(face_id) + '.' + str(count) + ".pgm", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 11: 
         break

print("\nExiting")
cam.release()
cv2.destroyAllWindows()

dataset_path = "faces"
file_list = filter_pgm(dataset_path)

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
    ])


Train = np.zeros((no_of_classes * 11 + 2, no_of_landmarks)) # Matrix of distances
train = 0
for subject in range(1, no_of_classes + 1):
    
    subject_landmarks = []
    if len(str(subject)) == 1:
        subject = "0" + str(subject)
    else:
        subject = str(subject)
        
    images_list = [x for x in file_list if re.search("^.*" + subject + ".*pgm$", x)]
    #images_list = [images_list[i] for i in indices]
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
        print(image_name)

        Train[train,:] = distances
        train += 1

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = [str(number) for number in range(1, face_id + 1)] 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = detector(img, 1)
    for i, face in enumerate(faces):
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  
        test_image = gray[y:y+h,x:x+w]
        
        faces = detector(test_image, 1)
        for i, face in enumerate(faces):
            shape = predictor(test_image, face)
            shape = face_utils.shape_to_np(shape)
                
        representative_landmark = [ sum([point[0] for point in shape]) / no_of_landmarks, sum([point[1] for point in shape]) / no_of_landmarks]
        test_image_distances = [0] * no_of_landmarks
        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            index = i
            for (x, y) in shape[i:j]:
                test_image_distances[index] = euclidean_distance(representative_landmark, (x,y))
                index += 1

        id = recognizer(Train, test_image_distances)
        
        if id in new_labels.keys():
            id = new_labels[id]
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    
    cv2.imshow('Camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
print("\n Exiting Program")
cam.release()
cv2.destroyAllWindows()

