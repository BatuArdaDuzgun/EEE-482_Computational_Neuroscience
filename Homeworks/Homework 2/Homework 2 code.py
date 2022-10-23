# -*- coding: utf-8 -*-

import sys # the primer already has it

import numpy as np # this brings simple matrix operations to python
import matplotlib.pyplot as plt # this brings nice loking plots to python

import scipy.io as sio # this is used to load MATLAB files
from scipy import signal as sig # this brings 2D convolution.
from mpl_toolkits import mplot3d # used for 3D plots

"""
If you want to run this code in a IDE you need to commend out the line below and write the line
question = '1'
then run the code for question 1 and write the line 
question = '2'
then run the code for question 2 
"""

question = sys.argv[1]

def Batu_Arda_Düzgün_21802633_hw2(question):
    if question == '1' :
        print("Answer of question 1.) \n")
        question1_part_a()        
        question1_part_b()
        question1_part_c()        

    elif question == '2' :
        print("Answer of question 2.) \n")
        question2_part_a()
        question2_part_b()
        question2_part_c()   
        question2_part_d()   
        question2_part_e()
        question2_part_f()


def question1_part_a():
    
    print("\nAnswer of question 1 part a: \n")
        
    data = sio.loadmat('c2p3.mat') # loading the data given to us
    counts = data['counts']
    stim = data['stim']
    
    averages = np.zeros( ( np.shape(stim)[0], np.shape(stim)[1] , 10 ) ) # will fill this with STA's for each of the 10 timesteps.

    for i in range(10): # calculating the STA's for the 10 time steps.
         
        averages[:,:, i ] = STA(stim, counts, i + 1 ) 
        
    for i in range(10):
        
        plt.figure()
        plt.imshow( averages[:,:, i] , cmap='gray', vmin=np.min(averages), vmax=np.max(averages))
        plt.colorbar()
        plt.title('STA: ' + str(i + 1) + ' Steps Before Spike')
        plt.yticks(np.arange(0, 16, step=1) , np.arange(1, 17, step=1))
        plt.xticks(np.arange(0, 16, step=1) , np.arange(1, 17, step=1))
        
    print("This LGN cell is selective for when the center of the image first darkens then instantly lights up and brightens. ")   
    
    
    
        
def STA(stim, counts, time_step): # the function to calculate STA for a given times step for a stimulus and spike count data pair. 
    total_sum = np.zeros( (np.shape(stim)[0], np.shape(stim)[1]) ) # we will add everything spike tringering stimulus to this. 
    
    for i in range (time_step, len(counts)): #looping trough the data set
        
        total_sum[:,:] += stim[:,:, i - time_step] * counts[i]
        
    spike_count = np.sum(counts[time_step:]) # finding total spike count
    average = total_sum / spike_count # taking the spike tringing stimules average.
    return average


        
def question1_part_b(): 
    
    print("\nAnswer of question 1 part b: \n")
    
    data = sio.loadmat('c2p3.mat') # loading the data given to us
    counts = data['counts']
    stim = data['stim']
    
    averages = np.zeros( ( np.shape(stim)[0], np.shape(stim)[1] , 10 ) ) # will fill this with STA's for each of the 10 timesteps.

    for i in range(10): # calculating the STA's for the 10 time steps.
         
        averages[:,:, i ] = STA(stim, counts, i + 1 ) 
    
    
    collum_summed_average = np.sum(averages, axis = 0) # the code which sums the values trough the collums of the matrix
    
    row_summed_average = np.sum(averages, axis = 1) # the code which sums the values trough the rows of the matrix
    
    plt.figure()
    plt.imshow( collum_summed_average.transpose() , cmap='gray' , origin='lower', vmin=np.min(row_summed_average), vmax=np.max(row_summed_average)) # I drew the transpose here to enphesize how we sum trough the collums of the matrix
    plt.colorbar()
    plt.title("collum summed STA's")
    plt.yticks(np.arange(0, 10, step=1) , np.arange(1, 11, step=1))
    plt.xticks(np.arange(0, 16, step=1) , np.arange(1, 17, step=1))
    plt.ylabel("times steps")
    
    
    plt.figure()
    plt.imshow( row_summed_average , cmap='gray' , origin='lower', vmin=np.min(row_summed_average), vmax=np.max(row_summed_average))
    plt.colorbar()
    plt.title("row summed STA's")
    plt.xticks(np.arange(0, 10, step=1) , np.arange(1, 11, step=1))
    plt.yticks(np.arange(0, 16, step=1) , np.arange(1, 17, step=1))
    plt.xlabel("times steps")
             
    print("Because the LGN cell wants a special order of changes (first darkness then brightness), this matrix is not space time separable.")

def question1_part_c(): 
    
    print("\nAnswer of question 1 part c: \n")

    data = sio.loadmat('c2p3.mat') # loading the data given to us
    counts = data['counts']
    stim = data['stim']
    
    averages = np.zeros( ( np.shape(stim)[0], np.shape(stim)[1] , 10 ) ) # will fill this with STA's for each of the 10 timesteps.

    for i in range(10): # calculating the STA's for the 10 time steps.
         
        averages[:,:, i ] = STA(stim, counts, i + 1 ) 
  

    stimulus_projected_on_STA = np.zeros(len(counts)) # we will fill this with the results of each stimuluses projection.
    for i in range(len(counts)):  # we look at every mesurement point
        for j in range( np.shape(averages)[1] ): # this for loop is for the Frobenius inner product calculation
            
            stimulus_projected_on_STA[i] += np.inner(averages[:,j,0], stim[:,j,i]) 

    stimulus_projected_on_STA = stimulus_projected_on_STA / np.max(np.abs(stimulus_projected_on_STA)) # making the maximum 1
    
    plt.figure(figsize=(12, 6)) # this lets me decide how big the plot should be
    plt.title("histogram of all stimulus projections")
    plt.ylabel("normalized spike counts")
    plt.xlabel("projected stimulus")
    plt.grid( alpha=.4) # this is the line which creats the grid line which increases the redability of the plot
    plt.ylim(0, 2) # this is here so all the graphs have the same y axis'
    plt.hist(stimulus_projected_on_STA, density = True , bins=100)
     
    
    nonzero_stimulus_projected_on_STA = [] # this is an empth list we will convert it to an array once it reaches its full size.
    for i in range(1 , len(counts)): # we look at every mesurement point
        
        if 0 != counts[i]: # look if there are any spikes
            
            inner_product = 0 #we will use this to calculate the inner product
            
            for j in range( np.shape(averages)[1] ): # this for loop is for the Frobenius inner product calculation
            
                inner_product += np.inner(averages[:,j,0], stim[:,j,i - 1])
            
            
            nonzero_stimulus_projected_on_STA.append(inner_product) # we append the summed total to the list
    
    nonzero_stimulus_projected_on_STA = np.array(nonzero_stimulus_projected_on_STA) # converting our list to an array
    
    nonzero_stimulus_projected_on_STA = nonzero_stimulus_projected_on_STA / np.max(np.abs(nonzero_stimulus_projected_on_STA)) # making the maximum 1
    
    plt.figure(figsize=(12, 6))  # this lets me decide how big the plot should be
    plt.title("histogram of stimulus projections at time bins where a non-zero spike counts accures")
    plt.ylabel("normalized spike counts")
    plt.xlabel("projected stimulus")
    plt.grid( alpha=.4) # this is the line which creats the grid line which increases the redability of the plot
    plt.ylim(0, 2) # this is here so all the graphs have the same y axis'
    plt.hist(nonzero_stimulus_projected_on_STA, bins=100, color = "green", density = True )  
    
    print("The STA doesn’t discrimination spike-eliciting stimuli very significantly because only around 80% of the Frobenius inner products’ result is bigger than 0. Still, we can say the STA discriminates spike-eliciting stimuli but not very significantly.")
    
    plt.figure(figsize=(12, 6))  # this lets me decide how big the plot should be
    plt.title("histogram of all stimulus projections and stimulus projections at time bins where a non-zero spike count accures")
    plt.ylabel("normalized spike counts")
    plt.xlabel("projected stimulus")
    plt.grid( alpha=.4) # this is the line which creats the grid line which increases the redability of the plot
    plt.ylim(0, 2) # this is here so all the graphs have the same y axis'
    plt.hist(stimulus_projected_on_STA, bins=100, label="all stimulus", density = True )
    plt.hist(nonzero_stimulus_projected_on_STA, bins=100, color = "green", alpha = 0.5, label="non-zero spike count stimulus", density = True )
    plt.legend(loc='upper right')
    plt.show() # this privents the plot from closing themselfs at the end     



def question2_part_a(): 
    
    print("\nAnswer of question 2 part a: \n")

    DOG_receptive_field = np.zeros((21 , 21)) # we will file this in

    for i in range(-10, 11): # creation of the 21 21 DOG field
        for j in range(-10, 11):
            
            DOG_receptive_field[i + 10 , j + 10] = DOG(j , i , 4 , 2) 
          
            
    plt.figure(figsize=(8, 8)) # code for showing the DOG filter
    plt.imshow( DOG_receptive_field , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("DOG receptive field")
    
    plt.figure(figsize=(8 , 8)) # code for showing the DOG filter in 3D
    ax = plt.axes(projection='3d')
    A = B = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(A, B)
    ax.plot_surface(X, Y, DOG_receptive_field, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('DOG receptive field')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('values of DOG receptive field')
    ax.view_init(elev= 15, azim=35) # where we see the 3D plot from
    
    print("DOG receptive field is a edge decetion filter")

def DOG (x, y, sigma_s, sigma_c): # the DOG function given in the HW guideline
    
    return 1/(2*np.pi*sigma_c**2) * np.exp(- (x**2+y**2) / (2*sigma_c**2)) - 1/(2*np.pi*sigma_s**2) * np.exp(- (x**2+y**2) / (2*sigma_s**2))



def question2_part_b(): 
    
    print("\nAnswer of question 2 part b: \n")
    
    monkey_img = plt.imread('hw2_image.bmp') # loading the image given to us in to Python     
       
    plt.figure(figsize=(8, 8)) # code for showing the DOG filter
    plt.imshow( monkey_img )
    plt.title("the original monkey image")

      
    DOG_receptive_field = np.zeros((21 , 21)) # we will file this in

    for i in range(-10, 11): # creation of the 21 21 DOG field
        for j in range(-10, 11):
            
            DOG_receptive_field[i + 10 , j + 10] = DOG(j , i , 4 , 2) 
        
    
    neural_response_img = sig.convolve(monkey_img[:, :, 0], DOG_receptive_field , mode="valid") # Puting a DOG filter on every singal pixel.

    plt.figure(figsize=(8, 8)) # code for showing the DOG filtered monkey image
    plt.imshow( neural_response_img , cmap = "gray")
    plt.title("monkey image after filtering with DOG receptive field") 
    
    print("Basically, speaking difference-of-gaussians (DOG) is an edge detecting filter on the image and it makes the edges on the image more prominent. ")

def question2_part_c(): 
    
    print("\nAnswer of question 2 part c: \n")

    monkey_img = plt.imread('hw2_image.bmp') # loading the image given to us in to Python

    for sigma_c in range(1, 5): # code of testing difrent DOF parameter, threshold combinations a
        for sigma_s in range(sigma_c + 1 , 6):
            for threshold in (0 , 2 , 5):
                
                draw_thresholded_dog_filtered_img(sigma_s, sigma_c, threshold, monkey_img)
                
                  
    print("\nI belive the best ones are sigma_s = 4, sigma_c = 1, threshold = 0 and sigma_s = 5, sigma_c = 1, threshold = 5 but the one with treash hold 5 is slightly better")
    

def thresholding_filter(threshold, img): # treash holding filter funtion, this will be re-used in part f

    for i in range(np.shape(img)[0]): # we loop on each pixel of the image to do a treashold cheak on all of them
        for j in range(np.shape(img)[1]):
            
            if img[i , j] >= threshold:
                
                img[i , j] = 1
                
            else:
                
                img[i , j] = 0
                
    return img


def draw_thresholded_dog_filtered_img(sigma_s, sigma_c, threshold, img):
    
    DOG_receptive_field = np.zeros((21 , 21)) # we will file this in

    for i in range(-10, 11): # creation of the 21 21 DOG field
        for j in range(-10, 11):
            
            DOG_receptive_field[i + 10 , j + 10] = DOG(j , i , sigma_s , sigma_c) 

    neural_response_img = sig.convolve(img[:, :, 0], DOG_receptive_field , mode="valid") # Puting the DOG filter on every singal pixel.
    
    filtered_img = thresholding_filter(threshold, neural_response_img)

    plt.figure(figsize=(8, 8)) # code for showing the DOG filtered edge detected monkey image
    plt.imshow(filtered_img , cmap = "gray")
    plt.title("monkey image after DOG filtering with sigma_s = " + str(sigma_s) + " and sigma_c = " + str(sigma_c) + "\nthen edge detection with thresholed = " + str(threshold)) 



def question2_part_d(): 
    
    print("\nAnswer of question 2 part d: \n")

    gabor_receptive_field = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field[i + 10 , j + 10] = gabor(np.array([j , i]) , np.pi/2 )        
        
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 90 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8 , 8)) # code for showing the gabor filter in 3D
    ax = plt.axes(projection='3d')
    A = B = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(A, B)
    ax.plot_surface(X, Y, gabor_receptive_field, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title("gabor receptive field for 90 degrees");
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel("values of gabor receptive field for 90 degrees")
    ax.view_init(elev=15, azim=35) # where we see the 3D plot from

    print("From this image we can see gabor filter detects edges with an orientation close to θ degrees and it makes these edges more prominent. ")



def gabor(x, theta): # the gabor function given in the HW guideline
    sigma_l = 3 # this time most of the internal parematers are taken as a constant because we are not asked to change them.
    sigma_w = 3
    lambda_ = 6
    phi = 0
    k_theta = np.array([np.cos(theta), np.sin(theta)])
    k_theta_ort = np.array([-np.sin(theta), np.cos(theta)])

    D = np.exp( -( (k_theta.dot(x))**2) / (2*(sigma_l**2))- (k_theta_ort.dot(x)**2) / (2*(sigma_w**2))) * np.cos(2 * np.pi * k_theta_ort.dot(x) / lambda_ + phi)

    return D



def question2_part_e(): 
    
    print("\nAnswer of question 2 part e: \n")
    
    monkey_img = plt.imread('hw2_image.bmp') # loading the image given to us in to Python

    
    gabor_receptive_field = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field[i + 10 , j + 10] = gabor(np.array([j , i]) , np.pi/2 ) 
    
    
    neural_response_img = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field , mode="valid") # Puting a DOG filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the DOG filtered edge detected monkey image
    plt.imshow(neural_response_img , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = pi/2")             

    print("From this image we can see gabor filter detects edges with an orientation close to θ degrees and it makes these edges more prominent.")



def question2_part_f(): 
    
    print("\nAnswer of question 2 part f: \n")

    monkey_img = plt.imread('hw2_image.bmp') # loading the image given to us in to Python
    
    gabor_receptive_field_pi_over_2 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_pi_over_2[i + 10 , j + 10] = gabor(np.array([j , i]) , np.pi/2 ) 
    
    neural_response_img_pi_over_2 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_pi_over_2 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_pi_over_2 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 90 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_pi_over_2 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = pi/2")        
   
    
    gabor_receptive_field_pi_over_3 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_pi_over_3[i + 10 , j + 10] = gabor(np.array([j , i]) , np.pi/3 ) 
    
    neural_response_img_pi_over_3 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_pi_over_3 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_pi_over_3 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 60 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_pi_over_3 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = pi/3")    
    
    
    gabor_receptive_field_pi_over_6 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_pi_over_6[i + 10 , j + 10] = gabor(np.array([j , i]) , np.pi/6 ) 
            
    neural_response_img_pi_over_6 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_pi_over_6 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_pi_over_6 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 30 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_pi_over_6 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = pi/6")    
          
    
    gabor_receptive_field_0 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_0[i + 10 , j + 10] = gabor(np.array([j , i]) , 0 )     
        
    neural_response_img_0 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_0 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_0 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 0 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_0 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = 0")    
        
    
    gabor_receptive_field_sum = neural_response_img_pi_over_2 + neural_response_img_pi_over_3 + neural_response_img_pi_over_6 + neural_response_img_0
    
    plt.figure(figsize=(8, 8)) # code for showing the sum of the gabor filtered monkey images
    plt.imshow(gabor_receptive_field_sum , cmap = "gray")
    plt.title("sum of 4 monkey images with 4 difrent angeled gabor filters") 

    # these are test done to improve the results

    gabor_receptive_field_sum_after_TH = thresholding_filter(150 , gabor_receptive_field_sum) # threshold filtering the result

    plt.figure(figsize=(8, 8)) # code for showing the sum of the gabor filtered monkey images after the sum is threshold filtered
    plt.imshow(gabor_receptive_field_sum_after_TH , cmap = "gray")
    plt.title("sum of 4 monkey images with 4 difrent angeled gabor filters \nafter it is threshold filtered") 


    print("threshold can be further improved by adding the missing 120 degrees and 150 degrees edges. So, I will add them.")


    gabor_receptive_field_2pi_over_3 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_2pi_over_3[i + 10 , j + 10] = gabor(np.array([j , i]) , 2*np.pi/3 ) 
    
    neural_response_img_2pi_over_3 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_2pi_over_3 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_2pi_over_3 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 120 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_2pi_over_3 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = 2pi/3") 
    
    
    gabor_receptive_field_5pi_over_6 = np.zeros((21 , 21)) # we will file this in
    
    for i in range(-10, 11): # creation of the 21 21 gabor field
        for j in range(-10, 11):
            
            gabor_receptive_field_5pi_over_6[i + 10 , j + 10] = gabor(np.array([j , i]) , 5*np.pi/6 ) 
    
    neural_response_img_5pi_over_6 = sig.convolve(monkey_img[:, :, 0], gabor_receptive_field_5pi_over_6 , mode="valid") # Puting a gabor filter on every singal pixel.
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filter
    plt.imshow( gabor_receptive_field_5pi_over_6 , origin='lower')
    plt.colorbar()
    plt.xticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.yticks(np.arange(0, 21, step=1) , np.arange(-10, 11, step=1))
    plt.title("gabor receptive field for 150 degrees")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    
    plt.figure(figsize=(8, 8)) # code for showing the gabor filtered monkey image
    plt.imshow(neural_response_img_5pi_over_6 , cmap = "gray")
    plt.title("monkey image after gabor filtering with theta = 5pi/6") 
    
    
    gabor_receptive_field_sum_2 = neural_response_img_pi_over_2 + neural_response_img_pi_over_3 + neural_response_img_pi_over_6 + neural_response_img_0 + neural_response_img_2pi_over_3 + neural_response_img_5pi_over_6

    plt.figure(figsize=(8, 8)) # code for showing the sum of the gabor filtered monkey images
    plt.imshow(gabor_receptive_field_sum_2 , cmap = "gray")
    plt.title("sum of 6 monkey images with 6 difrent angeled gabor filters") 

    # these are test done to improve the results

    gabor_receptive_field_sum_after_TH_2 = thresholding_filter(250 , gabor_receptive_field_sum_2) # threshold filtering the result

    plt.figure(figsize=(8, 8)) # code for showing the sum of the gabor filtered monkey images after the sum is threshold filtered
    plt.imshow(gabor_receptive_field_sum_after_TH_2 , cmap = "gray")
    plt.title("sum of 6 monkey images with 6 difrent angeled gabor filters \nafter it is threshold filtered") 
    plt.show() # this privents the plot from closing themselfs at the end 


    
Batu_Arda_Düzgün_21802633_hw2(question) # this is the only line which runs so it is very important.
    
    
    
    
    
    
    
    
    
    
