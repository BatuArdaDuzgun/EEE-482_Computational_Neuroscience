# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:10:38 2021

@author: mehme
"""
import os
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.image import index_img
from nilearn.image import load_img
from nilearn import masking
from nilearn.input_data import NiftiMasker



"""
MASK, SOM norm, Confusion matrix
"""

'''
Write the filepaths by changing \ to /
'''
#plotting.plot_img('C:/Users/mehme/EEE482_data/sub-1/anat/sub-1_T1w.nii.gz')



def parser():
    haxbydir = "C:/Users/a/Documents/Brain data for EEE 482 project/"
    haxbysubjects = os.listdir(haxbydir)
    timehorizon = np.arange(0,302.5,2.5)
    
    #reference arrays for dictionaryies
    haxbysubjects = haxbysubjects[4:10]
    inputs = ['face', 'cat', 'shoe', 'chair', 'scissors', 'bottle', 'house', 'scrambled']
    timehorizon = np.arange(0,302.5,2.5)
    
    #Parsing directory into dictionary for easier access
    haxby = {}
    for i in haxbysubjects:
        directory = os.listdir(haxbydir+i)
        haxby[i] = {"mask": haxbydir + i + '/'+directory[2],
                    'anat': haxbydir + i + '/'+directory[0]+'/'+(os.listdir(haxbydir+i+'/'+directory[0])[0]),
                    'fun': os.listdir(haxbydir + i + '/'+directory[1])}
    for i in (haxbysubjects):
        haxby[i]['data'] = {}
        for j in range(len(haxby[i]['fun'])):
            if haxby[i]['fun'][j].find('.tsv') == -1:
                continue
            else:
                behavior = pd.read_csv(haxbydir + i + '/'+'func/'+haxby[i]['fun'][j], sep='\t')
                conditions = np.array(['rest']*121)
                for abc in range(8):
                    conditions[(((15 + 36*abc)<timehorizon) & (timehorizon <= (15 + 36*abc + 20)))] = behavior ['trial_type'][abc*12]
                haxby[i]['data'][str(int((j+1)/2))+'_behavioral'] = conditions
                
        for j in range(len(haxby[i]['fun'])):
            if haxby[i]['fun'][j].find('.tsv') == -1:
                target_img = load_img(haxbydir + i + '/'+'func/'+haxby[i]['fun'][j]) #index_img(haxbydir + i + '/'+'func/'+haxby[i]['fun'][j], haxby[i]['data'][str(int((j+2)/2))+'_behavioral'])
                mask = haxby['sub-2']['mask']#masking.compute_brain_mask(target_img, threshold=0.5, connected=True, opening=2, verbose=0)
                masker = NiftiMasker(mask_img=mask,
                         smoothing_fwhm=4, standardize=True,
                         memory="nilearn_cache", memory_level=1)
                haxby[i]['data'][str(int((j+2)/2))+'_fmrimasked'] =  masker.fit_transform(target_img)
    haxby.pop('sub-5')
    haxbysubjects.pop(4)

    

    return haxby, haxbysubjects, inputs, masker

haxby, haxbysubjects, inputs, masker = parser()


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn_som.som import SOM
from scipy import stats


fmri_masked = haxby['sub-2']['data']['1_fmrimasked']
conditions = haxby['sub-2']['data']['1_behavioral']
session_label = np.zeros(121)

som = SOM(m=8, n=1, dim=464) # this is the model


from sklearn.feature_selection import RFE
#feature_selection = RFE(LinearSVC(penalty='l1', dual=False), 50, step=0.25) #LinearSVC(), 50, step=0.25
#classifier = OneVsRestClassifier( som )
#rfe_svc = OneVsRestClassifier( Pipeline([('rfe', feature_selection), ('svc', LinearSVC())]) )
fmri_masked = haxby['sub-2']['data']['1_fmrimasked']
cv = LeaveOneGroupOut()

for i in range(2,13):
    fmri2 = haxby['sub-2']['data'][str(i) + '_fmrimasked']
    fmri_masked = np.concatenate((fmri_masked, fmri2), axis=0)
    conditions = np.concatenate((conditions, haxby['sub-2']['data'][str(i) + '_behavioral']))
    group2 = np.ones(121)*(i-1)
    session_label  = np.concatenate((session_label, group2))
    
condition_mask = conditions != 'rest' #(conditions  == 'face') | (conditions  == 'cat') #conditions != 'rest'

conditions2 = conditions[condition_mask]
session_label2 = session_label[condition_mask]
fmri_masked2 = fmri_masked[condition_mask]


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # this brings nice loking plots to python



Confused = np.zeros((8, 8))



som.fit(fmri_masked2)

y_pred_ovo = som.predict(fmri_masked2)


maping = {}

for i in (list(dict(conditions2))):
    
    index = conditions2 == i
    
    maping[i] = stats.mode(y_pred_ovo(index))




"""
temp_matrix = confusion_matrix(y_pred_ovo, conditions2[session_label2 == i])

Confused += temp_matrix
"""

"""


Confused = y_pred_ovo


fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(Confused, cmap='YlGnBu' )

# We want to show all ticks...
ax.set_xticks(np.arange(len(inputs)))
ax.set_yticks(np.arange(len(inputs)))
# ... and label them with the respective list entries
ax.set_xticklabels(inputs)
ax.set_yticklabels(inputs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.title("Confusion Matrix for Ventral Temporal Cortex Mask \n and SVM with SOM Classifier")
plt.xlabel('True Class')
plt.ylabel('Predicted Class')

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)


for i in range(8):
    for j in range(8):
        if Confused[i, j] > 50:
            text = ax.text(j, i, Confused[i, j], ha="center", va="center", color="w")
        else:
            text = ax.text(j, i, Confused[i, j], ha="center", va="center", color="k")
        
        
"""    
        
        
        
        
        
        
        
        