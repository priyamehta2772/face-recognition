# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:26:47 2019

@author: LENOVO
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize(accuracy, labels):
    df = pd.DataFrame({'Train Images':list(accuracy.keys()),'Accuracy':list(accuracy.values())})
    plot = sns.barplot(x = "Train Images",y = "Accuracy",data = df ,hue_order = labels)
    plt.figure(figsize=(15,16)) 
    for item in plot.get_xticklabels():
        item.set_rotation(60)


#Visualizing Train images combination for Approach 1
''' 
    Accuracy                Train images combination
28.88888888888889 -> [2, 3, 1, 9, 5, 6, 0, 8, 7, 10, 4]
35.55555555555556 -> [3, 4, 5, 10, 6, 8, 9, 7, 2, 1, 0]
26.666666666666668 -> [8, 9, 1, 0, 10, 6, 5, 4, 7, 3, 2]
42.22222222222222 -> [9, 2, 3, 10, 5, 6, 0, 7, 8, 4, 1]
40.0 -> [10, 0, 5, 3, 6, 8, 9, 2, 1, 4, 7]
31.11111111111111 -> [5, 8, 1, 0, 3, 6, 2, 10, 9, 4, 7]
'''
accuracy = {"[2,3,1,9,5,6,0,8]":28.88, "[3, 4, 5, 10, 6, 8, 9, 7]":35.55, "[8, 9, 1, 0, 10, 6, 5, 4]":26.66, 
            "[9, 2, 3, 10, 5, 6, 0, 7]":42.22, "[10, 0, 5, 3, 6, 8, 9, 2]":40.0, "[5, 8, 1, 0, 3, 6, 2, 10]":31.11}
visualize(accuracy, labels)

# Visualizing Train images combination for Approach 2
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
accuracy = {0:88.88, 1:57.77, 2:93.33, 3:93.33, 4:55.55, 5:62.22, 6:91.11, 7:62.22}
visualize(accuracy)




