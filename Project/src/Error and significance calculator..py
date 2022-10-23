# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:20:10 2021

@author: a
"""
import numpy as np
import matplotlib.pyplot as plt # this brings nice loking plots to python
from scipy.stats import norm



Total_stimulus_number = 768 

RFE_SVM_l1 = 42 + 62 + 53 + 79 + 90 + 32 + 89 + 51
RFE_SVM_l2 = 55 + 60 + 52 + 86 + 90 + 39 + 86 + 54
RFE_LR_l1  = 66 + 63 + 52 + 89 + 95 + 48 + 91 + 67
RFE_LR_l2  = 75 + 78 + 57 + 90 + 96 + 52 + 94 + 71
RFE_LDA    = 43 + 52 + 26 + 72 + 87 + 35 + 81 + 41

Anova_SVM_l1 = 487
Anova_SVM_l2 = 484
Anova_LR_l1  = 485
Anova_LR_l2  = 482
Anova_LDA    = 55 + 50 + 41 + 74 + 92 + 33 + 92 + 55

Mask_SVM_l1 = 594
Mask_SVM_l2 = 623
Mask_LR_l1  = 605
Mask_LR_l2  = 623
Mask_LDA    = 41 + 50 + 54 + 58 + 81 + 44 + 68 + 50

Dummy = 96

succeses = np.array( [[RFE_SVM_l1, RFE_SVM_l2, RFE_LR_l1, RFE_LR_l2, RFE_LDA, Dummy] ,
          [Anova_SVM_l1, Anova_SVM_l2, Anova_LR_l1, Anova_LR_l2, Anova_LDA, Dummy] ,
          [Mask_SVM_l1, Mask_SVM_l2, Mask_LR_l1, Mask_LR_l2, Mask_LDA, Dummy] ] )

percent_succes = succeses / Total_stimulus_number * 100

percent_succes_nice = np.zeros((3, 6))

for i in range(3):
    for j in range(6):
        percent_succes_nice[i, j] = float(( "{:.2f}".format(percent_succes[i, j])))
        
    

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(percent_succes, cmap='RdYlGn' )

# We want to show all ticks...
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(3))
# ... and label them with the respective list entries
ax.set_xticklabels(("SVM l1" , "SVM l2" , "LR l1" , "LR l2" , "LDA" , "Dummy\nClassifier"))
ax.set_yticklabels(("RFE" , "Anova" , "Mask"))


plt.title("Success rates for Different Pipelines")
plt.xlabel('Clasifieres')
plt.ylabel('Preprocessors')


for i in range(3):
    for j in range(6):
        text = ax.text(j, i, str(percent_succes_nice[i, j]) + "%", ha="center", va="center", color="k") 
            
        
        
mean = 1 / 8 * Total_stimulus_number 
var = 1/8 * 7/8 * Total_stimulus_number 
std = var ** (0.5)
    
significance_value = norm.cdf(-(succeses - mean)/std)

effect_size = (succeses - mean)/std
