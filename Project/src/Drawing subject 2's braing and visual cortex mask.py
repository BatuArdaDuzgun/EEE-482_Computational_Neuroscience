# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:10:38 2021

@author: mehme
"""
import os
import numpy as np
import pandas as pd

from nilearn.image import index_img
from nilearn.image import load_img
from nilearn import masking
from nilearn.input_data import NiftiMasker
'''
Write the filepath here by changing \ to /
'''
#plotting.plot_img('C:/Users/mehme/EEE482_data/sub-1/anat/sub-1_T1w.nii.gz')

haxbydir = "C:/Users/a/Documents/Brain data for EEE 482 project/"
haxbysubjects = os.listdir(haxbydir)
timehorizon = np.arange(0,302.5,2.5)

#reference arrays for dictionaryies
haxbysubjects = haxbysubjects[4:10]
inputs = ['face', 'cat', 'shoe', 'chair', 'scissors', 'bottle', 'house', 'scrambled']


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
                conditions[(((12+36*abc)<timehorizon) & (timehorizon <= (12 + 36*abc + 20)))] = behavior ['trial_type'][abc*12]
            haxby[i]['data'][str(int((j+1)/2))+'_behavioral'] = conditions
    
    for j in range(len(haxby[i]['fun'])):
        if haxby[i]['fun'][j].find('.tsv') == -1:
            target_img = load_img(haxbydir + i + '/'+'func/'+haxby[i]['fun'][j]) #index_img(haxbydir + i + '/'+'func/'+haxby[i]['fun'][j], haxby[i]['data'][str(int((j+2)/2))+'_behavioral'])
            mask = masking.compute_brain_mask(target_img, threshold=0.5, connected=True, opening=2, verbose=0)
            masker = NiftiMasker(mask_img=mask,
                     smoothing_fwhm=4, standardize=True,
                     memory="nilearn_cache", memory_level=1)
            haxby[i]['data'][str(int((j+1)/2))+'_fmrimasked'] =  masker.fit_transform(target_img)
haxby.pop('sub-5')
haxbysubjects.pop(4)

'''
Now visualise the mask in Haxby dataset on anatomical barin image in the same dataset.
'''
from nilearn import plotting
#for i in haxbysubjects:
  
plotting.plot_img(haxby["sub-2"]['anat'])
plotting.plot_roi(haxby["sub-2"]['mask'], bg_img=haxby['sub-2']['anat'], cmap = 'Paired')