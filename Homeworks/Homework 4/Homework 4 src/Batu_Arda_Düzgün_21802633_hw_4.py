# -*- coding: utf-8 -*-

import sys # the primer already has it

import numpy as np # this brings simple matrix operations to python
import matplotlib.pyplot as plt # this brings nice loking plots to python

import scipy.io as sio # this is used to load MATLAB files


from sklearn.decomposition import PCA # PCA function

from sklearn.decomposition import FastICA # function given in the manual
from sklearn.decomposition import NMF # function given in the manual


"""
If you want to run this code in a IDE you need to commend out the line below and write the line
question = '1'
then run the code for question 1 and write the line 
question = '2'
then run the code for question 2 
"""

question = sys.argv[1]

def Batu_Arda_Düzgün_21802633_hw4(question):
    if question == '1' :
        print("Answer of question 1.) \n")
        question1_part_a()        
        question1_part_b()
        question1_part_c()
        question1_part_d()        

    elif question == '2' :
        print("Answer of question 2.) \n")
        question2_part_a()
        question2_part_b()
        question2_part_c()   
        question2_part_d()   
        question2_part_e()


def my_dispImArray(images, title): # I am not explaing this in detail because it is the function which is given with the instructions.
       
    width = round((np.shape(images)[1]) ** (0.5) )
    
    mn = np.shape(images) # Compute rows, cols
    m = mn[0]
    n = mn[1]
    height = int(n / width)
    
    display_rows = int(np.floor(m ** (0.5))) # Compute number of items to display
    display_cols = int(np.ceil(m / display_rows))
    
    pad = 1 # padding that will be used
    
    if np.min(images) < 0: # this is to make the non negative matricies look nice.
        
        display_array = -np.ones((pad + display_rows * (height + pad), pad + display_cols * (width + pad))); # we will fill this in with the images
    
    else:
        
        display_array = np.zeros((pad + display_rows * (height + pad), pad + display_cols * (width + pad))); # we will fill this in with the images
    
    curr_ex = 0
    
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            
            if curr_ex == m:
                
                break
    		
            max_val = max(abs(images[curr_ex, :]))
            
            display_array[pad + j * (height + pad) : pad + j * (height + pad) + height, pad + i * (width + pad) : pad + i * (width + pad) + width] = images[curr_ex, :].reshape(height, width).T / max_val
            
            curr_ex += 1
    
        if curr_ex == m:
        
            break
    
    # Display Image
    plt.figure(figsize=(10, 10))
    plt.imshow(display_array, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def question1_part_a():
    
    print("\nAnswer of question 1 part a: \n")
    
    data = sio.loadmat('matlab_hw4_data1') # loading the data given to us
    faces = data["faces"]
    
    my_dispImArray(faces[0:1,:], "Face 1") # to test if the images are loaded correctly
    my_dispImArray(faces[1:2,:], "Face 2")
    
    """
    for i in range(0, 2): # to test if the images are loaded correctly
        plt.figure()
        plt.imshow(faces[i,:].reshape(32, 32).T, cmap=plt.cm.gray)
        plt.title('Face ' + str(i+1))
    """
    
    pca = PCA(100)
    PCA_of_faces = pca.fit(faces) # calculating PCA 
    
    pca_explaind_proprtion_of_var = PCA_of_faces.explained_variance_ratio_ # the function already computs this for us
    
    plt.figure(figsize=(16,10)) #ploting Proportion of Variance Explained by each Individual PC
    plt.plot(range(1, 101), pca_explaind_proprtion_of_var, "o")
    plt.xlabel('Principal Component Index') 
    plt.ylabel('Proportion of the explaind variance')
    plt.title('Plot of the Proportion of Variance Explained by each Individual PC')
    plt.grid()
    
    pca_componets = PCA_of_faces.components_ # taking the principal components from the function
    
    my_dispImArray(pca_componets[0:25 , :], "The First 25 PCs of the Data using PCA") # displaying the first 25 PCs

    print("The faces seem to be loaded correctly and the PC's disturbution show these data can PCA decomposed.")

def question1_part_b(): 
        
    print("\nAnswer of question 1 part b: \n")
    
    data = sio.loadmat('matlab_hw4_data1') # loading the data given to us
    faces = data["faces"]
        
    pca = PCA(100)
    PCA_of_faces = pca.fit(faces) # calculating PCA 
    
    pca_componets = PCA_of_faces.components_  # taking the principlal components. this part is repeated
    
    
    
    pca_projection_10 = faces.dot(pca_componets[0:10].T) # lower dimensionlar reprisentation for 10 PCs
    
    Reconstructed_faces_10 = pca_projection_10.dot(pca_componets[0:10]) # Reconstructed faces for 10 PCs
    
    my_dispImArray(faces[0:36], "The Original Images")
    
    my_dispImArray(Reconstructed_faces_10[0: 36], "The Reconstructed Images from the First 10 PCs")
    
    
    pca_projection_25 = faces.dot(pca_componets[0:25].T) # lower dimensionlar reprisentation for 25 PCs
    
    Reconstructed_faces_25 = pca_projection_25.dot(pca_componets[0:25]) # Reconstructed faces for 25 PCs
    
    my_dispImArray(Reconstructed_faces_25[0: 36], "The Reconstructed Images from the First 25 PCs")
    
    
    pca_projection_50 = faces.dot(pca_componets[0:50].T) # lower dimensionlar reprisentation for 50 PCs
    
    Reconstructed_faces_50 = pca_projection_50.dot(pca_componets[0:50]) # Reconstructed faces for 50 PCs
    
    my_dispImArray(Reconstructed_faces_50[0: 36], "The Reconstructed Images from the First 50 PCs")
    
    
    
    
    MSE_10 = (Reconstructed_faces_10 - faces) ** 2 # calculating MSE for 10 PCs
    
    mean_10 = np.mean( np.mean(MSE_10 , 1) )# calculating the mean of MSE for 10 PCs
    STD_10 = np.std( np.mean(MSE_10 , 1) )# calculating the std of MSE for 10 PCs
    
    print("Mean of MSE for the first 10 PCs: " + str(mean_10)) #displaying the results
    print("STD of MSE for the first 10 PCs: " + str(STD_10) + "\n")
    
    
    MSE_25 = (Reconstructed_faces_25 - faces) ** 2 # calculating MSE for 25 PCs
    
    mean_25 = np.mean( np.mean(MSE_25 , 1) )# calculating the mean of MSE for 25 PCs
    STD_25 = np.std( np.mean(MSE_25 , 1) )# calculating the std of MSE for 25 PCs
    
    print("Mean of MSE for the first 25 PCs: " + str(mean_25)) #displaying the results
    print("STD of MSE for the first 25 PCs: " + str(STD_25) + "\n")
    
    
    MSE_50 = (Reconstructed_faces_50 - faces) ** 2 # calculating MSE for 50 PCs
    
    mean_50 = np.mean( np.mean(MSE_50 , 1) )# calculating the mean of MSE for 50 PCs
    STD_50 = np.std( np.mean(MSE_50 , 1) )# calculating the std of MSE for 50 PCs
    
    print("Mean of MSE for the first 50 PCs: " + str(mean_50)) #displaying the results
    print("STD of MSE for the first 50 PCs: " + str(STD_50) + "\n")
  
    print("We can see that, as the number of PCs used to reconstruct the images increases, the images look more like the originals. This is not surprising as increasing number of PCs means storing more detailed information in the low dimensional representation and images which are created with more information are detailed and look better ")
 
   
 

 
def question1_part_c(): # the proper one
    
    print("\nAnswer of question 1 part c: \n")

    print("Every time this code is run the result will be slitly different because FastICA function is not deterministic.")

    data = sio.loadmat('matlab_hw4_data1') # loading the data given to us
    faces = data["faces"]
    
    
    ICA_model_10 = FastICA(50, max_iter=10000) # data used for the transformation, it seemslike the only number which we enter is the lasteig
    ICA_of_faces_10 = ICA_model_10.fit(faces.T) # data used for the transformation fit to our face data
    
    
    components_10 = ICA_model_10.fit(faces.T).transform(faces.T)[:,0:10]# indipendent components of the model. we pick the first 10
    
    my_dispImArray(components_10.T, "The First 10 ICs of the Data using ICA") # displaying the ICs
    
    
    ICA_model_25 = FastICA(50, max_iter=10000) # data used for the transformation
    ICA_of_faces_25 = ICA_model_25.fit(faces.T) # data used for the transformation fit to our face data
    
    
    components_25 = ICA_model_25.fit(faces.T).transform(faces.T)[:,0:25]# indipendent components of the model.
    
    my_dispImArray(components_25.T, "The First 25 ICs of the Data using ICA") # displaying the ICs
    
    
    ICA_model_50 = FastICA(50, max_iter=10000) # data used for the transformation
    ICA_of_faces_50 = ICA_model_50.fit(faces.T) # data used for the transformation fit to our face data
    
    components_50 = ICA_model_50.fit(faces.T).transform(faces.T)[:,0:50]# indipendent components of the model.
    
    my_dispImArray(components_50.T, "The First 50 ICs of the Data using ICA") # displaying the ICs
    
    
    
    my_dispImArray(faces[0:36], "The Original Images") # displaying the original faces
    
    
    
    Reconstructed_faces_10 = components_10.dot((ICA_model_10.mixing_[:,0:10]).T) + ICA_model_10.mean_ # The function de-means each sample seperatly an we need to add it back.
    
    Reconstructed_faces_10 = Reconstructed_faces_10.T
    
    my_dispImArray(Reconstructed_faces_10[0:36,:], "The Reconstruct Images from the First 10 ICs") #displaying the reconstructed images
       
    
    Reconstructed_faces_25 = components_25.dot((ICA_model_25.mixing_[:,0:25]).T) + ICA_model_25.mean_ # The function de-means each sample seperatly an we need to add it back.
    
    Reconstructed_faces_25 = Reconstructed_faces_25.T
    
    my_dispImArray(Reconstructed_faces_25[0:36,:], "The Reconstruct Images from the First 25 ICs") #displaying the reconstructed images
    
      
    Reconstructed_faces_50 = components_50.dot(ICA_model_50.mixing_.T) + ICA_model_50.mean_ # The function de-means each sample seperatly an we need to add it back.
    
    Reconstructed_faces_50 = Reconstructed_faces_50.T
     
    my_dispImArray(Reconstructed_faces_50[0:36,:], "The Reconstruct Images from the First 50 ICs") #displaying the reconstructed images
    


    MSE_10 = (Reconstructed_faces_10 - faces) ** 2 # calculating MSE 
    
    mean_10 = np.mean( np.mean(MSE_10 , 1) )# calculating the mean
    STD_10 = np.std( np.mean(MSE_10 , 1) )# calculating the std
    
    print("Mean of MSE for the first 10 ICs: " + str(mean_10)) #displaying the results
    print("STD of MSE for the first 10 ICs: " + str(STD_10) + "\n")
  
    
    MSE_25 = (Reconstructed_faces_25 - faces) ** 2 # calculating MSE 
    
    mean_25 = np.mean( np.mean(MSE_25 , 1) )# calculating the mean
    STD_25 = np.std( np.mean(MSE_25 , 1) )# calculating the std
    
    print("Mean of MSE for the first 25 ICs: " + str(mean_25)) #displaying the results
    print("STD of MSE for the first 25 ICs: " + str(STD_25) + "\n")
  
    
    MSE_50 = (Reconstructed_faces_50 - faces) ** 2 # calculating MSE 
    
    mean_50 = np.mean( np.mean(MSE_50 , 1) )# calculating the mean
    STD_50 = np.std( np.mean(MSE_50 , 1) )# calculating the std
    
    print("Mean of MSE for the first 50 ICs: " + str(mean_50)) #displaying the results
    print("STD of MSE for the first 50 ICs: " + str(STD_50) + "\n")

    print("This method is very slitly better then PCA for the 50 component case. However, for the 10 and 25 component cases it is worse then the PCA, the reason for that is the lastEig value.")


def question1_part_d(): 

    print("\nAnswer of question 1 part d: \n")
    
    print("Every time this code is run the result will be slitly different because NMF function is not deterministic.")
        
    data = sio.loadmat('matlab_hw4_data1') # loading the data given to us
    faces = data["faces"]
    
    min_number_to_add = np.abs(np.min(faces)) # the most negative number in the faces matrix
    
    positive_faces = faces + min_number_to_add # prepairing for the NNMF 
    
    
    nnfm_model_10 = NMF(n_components=10,solver="mu", max_iter=1000) # model creation
    
    W_10 = nnfm_model_10.fit(positive_faces).transform(positive_faces) # these are the down samples faces.
    H_10 = nnfm_model_10.components_ # these are the MFs
    
    my_dispImArray(H_10, "The 10 MFs of the Data using NNMF") # displaying the 10 MFs
    
    
    nnfm_model_25 = NMF(n_components=25,solver="mu", max_iter=1000) # model creation
    
    W_25 = nnfm_model_25.fit(positive_faces).transform(positive_faces) # these are the down samples faces.
    H_25 = nnfm_model_25.components_ # these are the MFs
    
    my_dispImArray(H_25, "The 25 MFs of the Data using NNMF") # displaying the 25 MFs
    
    
    nnfm_model_50 = NMF(n_components=50, solver="mu", max_iter=1000) # model creation
    
    W_50 = nnfm_model_50.fit(positive_faces).transform(positive_faces) # these are the down samples faces.
    H_50 = nnfm_model_50.components_ # these are the MFs
    
    my_dispImArray(H_50, "The 50 MFs of the Data using NNMF") # displaying the 50 MFs
    


    my_dispImArray(faces[0:36], "The Original Images") # displaying the original faces


    Reconstructed_faces_10 = W_10.dot(H_10) - min_number_to_add# reconstracting the original faces 
    
    my_dispImArray(Reconstructed_faces_10[0:36, :], "The Reconstructed Images from the 10 MFs") # displaying the reconstrancted images.
    


    Reconstructed_faces_25 = W_25.dot(H_25) - min_number_to_add# reconstracting the original faces 
    
    my_dispImArray(Reconstructed_faces_25[0:36, :], "The Reconstructed Images from the 25 MFs") # displaying the reconstrancted images.
    


    Reconstructed_faces_50 = W_50.dot(H_50) - min_number_to_add# reconstracting the original faces 
    
    my_dispImArray(Reconstructed_faces_50[0:36, :], "The Reconstructed Images from the 50 MFs") # displaying the reconstrancted images.
    



    MSE_10 = (Reconstructed_faces_10 - faces) ** 2 # calculating MSE 
    
    mean_10 = np.mean( np.mean(MSE_10 , 1) )# calculating the mean
    STD_10 = np.std( np.mean(MSE_10 , 1) )# calculating the std
    
    print("Mean of MSE for the first 10 PCs: " + str(mean_10)) #displaying the results
    print("STD of MSE for the first 10 PCs: " + str(STD_10) + "\n")


    MSE_25 = (Reconstructed_faces_25 - faces) ** 2 # calculating MSE 
    
    mean_25 = np.mean( np.mean(MSE_25 , 1) )# calculating the mean
    STD_25 = np.std( np.mean(MSE_25 , 1) )# calculating the std
    
    print("Mean of MSE for the first 25 PCs: " + str(mean_25)) #displaying the results
    print("STD of MSE for the first 25 PCs: " + str(STD_25) + "\n")


    MSE_50 = (Reconstructed_faces_50 - faces) ** 2 # calculating MSE 
        
    mean_50 = np.mean( np.mean(MSE_50 , 1) )# calculating the mean
    STD_50 = np.std( np.mean(MSE_50 , 1) )# calculating the std
    
    print("Mean of MSE for the first 50 PCs: " + str(mean_50)) #displaying the results
    print("STD of MSE for the first 50 PCs: " + str(STD_50) + "\n")
    
    print("This is generating more error then the PCA because it has an aditional constraint of trying to keep every part of it none-negative. This is more realistic interms of the processes going inside the brain but it still gives worse results in term of error.")

    plt.show() # this privents the plot from closing themselves at the end





def neuron_response(x, mu_i, sigma = 1): # a single neuron
    
    result = 1 * np.exp(-((x - mu_i) ** 2) / (2 * (sigma ** 2)))
    
    return result


def question2_part_a():  
    
    print("\nAnswer of question 2 part a: \n")
    
    x = np.linspace(-16, 16, 1000) # different inputs to test the neurons
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    
    plt.figure(figsize=(16,10)) # the plot of neurons tuning curves
    
    for i in neurons:
        
        plt.plot(x, neuron_response(x, i))
    
    plt.grid()
    plt.xlabel('Stimulus')
    plt.ylabel('Responses')
    plt.title('21 Neurons Response to Stimulus')
    
    
    plt.figure(figsize=(16,10)) # the plot of neurons response to the input -1
    
    plt.plot(neurons, neuron_response(-1, neurons), "o")
    plt.grid()
    plt.xlabel("Neuron's prefered stimulus")
    plt.ylabel("Neuron's response to -1 stimulus")
    plt.title('21 Neurons Response to the -1 Stimulus')

    print("The tuning curves looks good, so we can do the rest of the question.")


def wta_decoder(neurons, responses):

    result = neurons[np.argmax(responses)] # finding the neuron which gives the highest response.
    
    return result


def trials(neurons, sigma, trial_count ):
    
    stimulus = []
    responses = []

    for i in range(trial_count):
        
        noise = np.random.normal(0, 1/20, len(neurons))
        
        stimulus.append((np.random.random(size = 1) - 0.5) * 10)
        
        response_without_noise = neuron_response(stimulus[i], neurons, sigma)
        
        response = response_without_noise + noise
        
        responses.append(response)
        
    return stimulus, responses




def question2_part_b(): 
    
    print("\nAnswer of question 2 part b: \n")
    
    print("Every time this code is run the result will be slitly different because trials function is random. \n")

    
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    stimulus, responses = trials(neurons, 1, 200) # the generated trials
    
    wta_estimate = [] # thses will be fild in
    wta_error = []
    
    for i in range(200): # looping throgh the triales
        
        wta_estimate.append(wta_decoder(neurons, responses[i])) # using the decoder
        
        wta_error.append(np.abs(wta_estimate[i] - stimulus[i])) # calculating the error of each trial   
        
    
    plt.figure(figsize=(16,10)) # ploting the actual and estimated stimulus on the same graph
    plt.plot(range(200), stimulus, "o")
    plt.plot(range(200), wta_estimate, "x")
    plt.legend(['Actual stimusul', 'WTA estimatin',])
    plt.ylabel('the stimulus value')
    plt.xlabel('the trial number')
    plt.title(' The Actual and WTA Estimated Stimulus')
    plt.grid()
    
    
    mean = np.mean( wta_error )# calculating the mean
    STD = np.std( wta_error )# calculating the std
    
    print("Mean of error for WTA estimation: " + str(mean)) #displaying the results
    print("STD of error for WTA estimation: " + str(STD) + "\n")

    print("WTA is a very crude method and because of it is creating th largest error")


def ml_decoder(neurons, responses, sigma_of_neurons):
    
    posible_x = np.arange(-6, 6, 1/200) # the function for sum of f(x_i) can not be solved linearly so we will test difrent x values
    
    
    log_likelyhoods = [] # these are not actualy the diferences from the perfect x but if x is perfrect this will give 0. because of this it is named like that
    
    
    for x in posible_x: # we find how well each x value minimizes the logarithmic log-likelihood function.
        
        log_likelyhood = sum((responses - neuron_response(x, neurons, sigma_of_neurons) ) ** 2) # I found this equation by hand
    
        log_likelyhoods.append(log_likelyhood)
        
    index_of_best_x = np.argmin(log_likelyhoods) # finding the index of best x
    best_x = posible_x[index_of_best_x] # finding the best x

    return best_x 


def question2_part_c(): 
    
    print("\nAnswer of question 2 part c: \n")
    
    print("Every time this code is run the result will be slitly different because trials function is random. \n")
    
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    stimulus, responses = trials(neurons, 1, 200)
    
    ml_estimate = []
    ml_error = []
    
    for i in range(200):
        
        ml_estimate.append(ml_decoder(neurons, responses[i], 1)) # using the decoder
        
        ml_error.append(np.abs(ml_estimate[i] - stimulus[i])) # calculating the error of each trial   
        
        
    
    plt.figure(figsize=(16,10)) # ploting the actual and estimated stimulus on the same graph
    plt.plot(range(200), stimulus, "o")
    plt.plot(range(200), ml_estimate, "x")
    plt.legend(['Actual stimusul', 'ML estimatin',])
    plt.ylabel('the stimulus value')
    plt.xlabel('the trial number')
    plt.title(' The Actual and ML Estimated Stimulus')
    plt.grid()
    
    
    mean = np.mean( ml_error )# calculating the mean
    STD = np.std( ml_error )# calculating the std
    
    print("Mean of error for ML estimation: " + str(mean)) #displaying the results
    print("STD of error for ML estimation: " + str(STD) + "\n")
    
    print("ML gives the over all best resluts, It doesn't use any prior knowlage.")
    

def map_decoder(neurons, responses, sigma_of_neurons):
    
    posible_x = np.arange(-6, 6, 1/200) # the function for sum of f(x_i) can not be solved linearly so we will test difrent x values
    
    log_likelyhoods = [] # these are not actualy the diferences from the perfect x but if x is perfrect this will give 0. because of this it is named like that
    
    for x in posible_x: # we find how well each x value minimizes the logarithmic log-likelihood function.
        
        log_likelyhood = sum(((responses - neuron_response(x, neurons, sigma_of_neurons) ) ** 2) / (2 * ((1/20) ** 2))) + ( x ** 2 ) / (2 * ((2.5) ** 2)) # I found this equation by hand
    
        log_likelyhoods.append(log_likelyhood) 
        
    index_of_best_x = np.argmin(log_likelyhoods)
    best_x = posible_x[index_of_best_x]

    return best_x




def question2_part_d(): 
    
    print("\nAnswer of question 2 part d: \n")
    
    print("Every time this code is run the result will be slitly different because trials function is random. \n")
    
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    stimulus, responses = trials(neurons, 1, 200)
    
    map_estimate = []
    map_error = []
    
    
    
    for i in range(200):
        
        map_estimate.append(map_decoder(neurons, responses[i], 1))
        
        map_error.append(np.abs(map_estimate[i] - stimulus[i]))
        
        
    
    plt.figure(figsize=(16,10))
    plt.plot(range(200), stimulus, "o")
    plt.plot(range(200), map_estimate, "x")
    plt.legend(['Actual stimusul', 'MAP estimatin',])
    plt.ylabel('the stimulus value')
    plt.xlabel('the trial number')
    plt.title(' The Actual and MAP Estimated Stimulus')
    plt.grid()
    
    
    mean = np.mean( map_error )# calculating the mean
    STD = np.std( map_error )# calculating the std
    
    print("Mean of error for MAP estimation: " + str(mean)) #displaying the results
    print("STD of error for MAP estimation: " + str(STD) + "\n")
    
    print("MAP gives a slitly worse reslut then the ML method, but it does use the prior knowlage.")


def question2_part_e(): 
    
    print("\nAnswer of question 2 part e: \n")
    
    print("Every time this code is run the result will be slitly different because trials function is random. \n")

    test_sigmas = [0.1, 0.2, 0.5, 1, 2, 5] # given in the manual.
    
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    for sigma in test_sigmas: # basicly doing part c for each sigma value
        
        stimulus, responses = trials(neurons, sigma, 200)
        
        ml_estimate = [] # will fill with new values on each loop
        ml_error = []
        
        for i in range(200):
            
            ml_estimate.append(ml_decoder(neurons, responses[i], sigma)) # using the ML decoder             
            
            ml_error.append(np.abs(ml_estimate[i] - stimulus[i])) # finding the error

              
        ml_mean = np.mean( ml_error )# calculating the mean
        ml_STD = np.std( ml_error )# calculating the std
        
        print("Mean of error for ML estimation with sigma_i = " + str(sigma) + " : " + str(ml_mean)) #displaying the results
        print("STD of error for ML estimation with sigma_i = " + str(sigma) + " : " + str(ml_STD) + "\n")

    
    # this is for explanation of the error and not part of the asked plot.

    x = np.linspace(-16, 16, 3200) # different inputs to test the neurons
    neurons = np.arange(-10, 11, 1) # the 21 neurons
    
    
    plt.figure(figsize=(16,10)) # the plot of neurons tuning curves
    
    for i in neurons:
        
        plt.plot(x, neuron_response(x, i, 0.1))
    
    plt.grid()
    plt.xlabel('Stimulus')
    plt.ylabel('Responses')
    plt.title('21 Neurons Response to Stimulus for sigma_i = 0.1 \n(This is to explain the error)')
    
    print("There is a sweet spot for the tuning curve with and for this setup it seems to be between 0.5 and 1. If we can not have the best tuning curve value it is a lot better to have a too wide tubing curve compared to a too narrow tuning curve.")    
         
    plt.show() # this privents the plot from closing themselves at the end


Batu_Arda_Düzgün_21802633_hw4(question) # this is the only line which runs so it is very important.

