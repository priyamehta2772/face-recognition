# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:26:47 2019

@author: LENOVO
"""
import pandas as pd
import seaborn as sns

def visualize(accuracy):
    df = pd.DataFrame({'Train Images':list(accuracy.keys()),'Accuracy':list(accuracy.values())})
    sns.barplot(x = "Train Images",y = "Accuracy",data = df)



#Visualizing Train images combination for Approach 1
accuracy = {"Set 1":28.88, "Set 2":28.88, "Set 3":28.88, "Set 4":33.33, "Set 5":20.0, "Set 6":22.22, "Set 7":26.66, "Set 7":24.44}
visualize(accuracy)

# Visualizing Train images combination for Approach 2
"""
    Accuracy               Train images combination
88.88888888888889 -> [4, 1, 3, 0, 2, 5, 9, 8, 7, 10, 6]
57.77777777777778 -> [3, 4, 5, 1, 2, 6, 7, 8, 10, 0, 9]  
93.33333333333333 -> [0, 3, 7, 1, 10, 9, 6, 4, 8, 2, 5]   10,9,7,4,1,0
93.33333333333333 -> [2, 0, 8, 7, 10, 4, 9, 1, 5, 6, 3]
55.55555555555556 -> [8, 0, 4, 1, 7, 10, 2, 5, 9, 6, 3]
62.22222222222222 -> [4, 7, 1, 5, 10, 0, 8, 2, 3, 6, 9]
91.11111111111111 -> [0, 9, 6, 8, 1, 5, 4, 10, 3, 7, 2]
62.22222222222222 -> [3, 10, 4, 8, 0, 7, 5, 2, 9, 1, 6]
"""
accuracy = {"Set 1":88.88, "Set 2":57.77, "Set 3":93.33, "Set 4":93.33, "Set 5":55.55, "Set 6":62.22, "Set 7":91.11, "Set 8":62.22}
visualize(accuracy)

# Visualizing Train images combination for Approach 3
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
accuracy = {"Set 1":93.33, "Set 2":60.0, "Set 3":93.33, "Set 4":95.55, "Set 5":62.22, "Set 6":66.66, "Set 7":91.11, "Set 8":68.88}
visualize(accuracy)



