# -*- coding: utf-8 -*-

import sys # the primer already has it

import numpy as np # this brings simple matrix operations to python
import matplotlib.pyplot as plt # this brings nice loking plots to python

from scipy.stats import norm # used to calculet CDF's used to calculate the p value
import scipy.io as sio # this is used to load MATLAB files

"""
If you want to run this code in a IDE you need to commend out the line below and write the line
question = '1'
then run the code for question 1 and write the line 
question = '2'
then run the code for question 2 
"""

question = sys.argv[1]

def Batu_Arda_Düzgün_21802633_hw3(question):
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




def ridge_regression(X, y, lamda): 
    return np.linalg.inv(X.T.dot(X) + lamda * np.identity(np.shape(X)[1])).dot(X.T).dot(y) # matrix operation which gives the weight vector for the ridge regesion.



def cross_validationation(X, y, K, lamda):       
        
    fold_size = int(np.size(y)/K) # size of test and validationation 
      
    
    validation_R2 = 0 # different R^2 will be summed up in here and then will be devided by K to find the average  
    test_R2 = 0    
    
    for i in range(K):
        
        train_data_ind, test_data_ind, validation_data_ind = [], [], []
        
        test_data_start = i * fold_size # stating point of validation set for the given K
        validation_data_start = (i+1) * fold_size # stating point of test set for the given K
        train_data_start = (i+2) * fold_size # stating point of training set for the given K
        
        for j in range(2 * np.size(y)): # deciding which set every index in (0 , 999) will go to.
            
            if j in range(test_data_start, validation_data_start):
            
                test_data_ind.append(j % np.size(y))
        
            if j in range(validation_data_start, train_data_start):
                
                validation_data_ind.append(j % np.size(y))
        
            if  j in range(train_data_start, test_data_start + np.size(y)):
            
                train_data_ind.append(j % np.size(y))
                
                
        x_validation, x_test, x_training = X[validation_data_ind], X[test_data_ind], X[train_data_ind] # palcing every input output pare in their respective set
        
        y_validation, y_test, y_training = y[validation_data_ind], y[test_data_ind], y[train_data_ind]
              
    
        validation_weight = ridge_regression(x_training , y_training , lamda) # riged rigresion to find the weight of the input parameters  
        
        validation_R2 += (np.corrcoef(y_validation.transpose(), (x_validation.dot(validation_weight)).transpose() )[0, 1]) ** 2 # corrcoef returns the coralation coaficent matrix of its inputs so here its outpıts [0 1] is the coralation coaficent coaficent betwen theh real y and the y we calculated form the fit model. we takes its square to calculate the R^2 of the model
    
    
        test_weight = ridge_regression(np.concatenate((x_validation, x_training) , axis=0) , np.concatenate((y_validation, y_training) , axis=0) , lamda) # riged rigresion to find the weight of the input parameters for both the traing and the validation
    
        test_R2 += (np.corrcoef( y_test.transpose() , (x_test.dot(test_weight)).transpose() )[0, 1]) ** 2 # again finding R^2 for the test.
    
    
    validation_R2 /= K # taking the R^2 average
    test_R2 /= K

    return validation_R2, test_R2


    

def question1_part_a():
    
    print("\nAnswer of question 1 part a: \n")
    print("every time this code is run part b and part c of this code will give a Slightly different output because of the randomness in bootstraping \n ")
    print("this will take some time to compute because I am testing for 1250 different lambda values \n")
    
    data = sio.loadmat('matlab_HW3_data2') # loading the data given to us
    Xn = data["Xn"]
    Yn = data["Yn"]
        
    lamda_values = np.logspace(-3, 12, num=1250, base=10) # different lambda values to calculate proportion of explained variance
    
    validation_R2_for_lamda_values = [] # we will fill these with the R^2 values for all the lambda
    test_R2_for_lamda_values = []
    
    
    for lamda in lamda_values: # doing the K fold regresion for each of the lambda values
        
        validation_R2 , test_R2 = cross_validationation(Xn, Yn, 10, lamda)
        
        validation_R2_for_lamda_values.append(validation_R2)
        test_R2_for_lamda_values.append(test_R2)
        
        
    plt.figure(figsize=(12, 8)) # plotting R^2 curves
    plt.plot(lamda_values, test_R2_for_lamda_values)
    plt.plot(lamda_values, validation_R2_for_lamda_values)
    plt.legend(['Test', 'Validation',])
    plt.ylabel('proportion of explained variance R^2')
    plt.xlabel('ridge parameter lambda')
    plt.title('R^2 vs lambda')
    plt.xscale('log')
    plt.grid()
 
    
    max_R2 =  max(validation_R2_for_lamda_values)
    
    lamda_opt_index = validation_R2_for_lamda_values.index(max_R2)
    
    lamda_opt = lamda_values[lamda_opt_index]
    
    performance_of_lambda_opt = test_R2_for_lamda_values[lamda_opt_index] # finding the model performance for the optimum lambda
    
    print('optimal ridge parameter across cross-validation folds, measured on the validation set =' + str(lamda_opt) + " and the model with this lambda, gives a model performance of R^2 = " + str(performance_of_lambda_opt))
    

    
def bootstrap(iteration_count, X, y, lamda): # the function to generate input weights for n bootstraps for a given input, output and ridged regreasion 
    
    weights = [] # we will fill this in
    
    for i in range(iteration_count):
        
        bootstrap_indexs = np.random.choice(np.size(y), np.size(y)) # chosing the index of the input output pairs which will be added to this iteration
        
        X_iteration = X[bootstrap_indexs]  # adding the choosen input output pairs in to the iteration
        y_iteration = y[bootstrap_indexs]
                
        weights.append(ridge_regression(X_iteration, y_iteration, lamda)) # adding the weights for this iteration to the output list 
        
    return weights


        
def question1_part_b(): 
    
    print("\nAnswer of question 1 part b: \n")
    
      
    data = sio.loadmat('matlab_HW3_data2') # loading the data given to us
    Xn = data["Xn"]
    Yn = data["Yn"]
    
    weights = np.array(bootstrap(500, Xn, Yn, 0)) # getting the weights form 500 bootstrap iterations
    
    plt.figure(figsize=(8, 4)) # we know this distribution should look like a gausian but I wanted to show it looks like a gausian too.
    plt.title('the first weight parameters distribution for lambda = 0')
    plt.xlabel('w_1')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(weights[:,0], bins=19, density=True)
    
    plt.figure(figsize=(8, 4)) # we know this distribution should look like a gausian but I wanted to show it looks like a gausian too.
    plt.title('the 4th weight parameters distribution for lambda = 0')
    plt.xlabel('w_4')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(weights[:,3], bins=19, density=True)  
    
    plt.figure(figsize=(8, 4)) # we know this distribution should look like a gausian but I wanted to show it looks like a gausian too.
    plt.title('the 8th weight parameters distribution for lambda = 0')
    plt.xlabel('w_8')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(weights[:,7], bins=19, density=True)    
    
    
    weights_mean = np.mean(weights, axis=0) # calculating the mean and standart deviation of each of the 100 weights
    weights_std = np.std(weights, axis=0)
    
     
    _95_percent_confidence = 2 * np.concatenate((weights_std.transpose(),weights_std.transpose()) , axis = 0) # the weights are normal distribution so 2 standart deviations to the both sides equals 95% confidence interval.
    
    
    p_values = 2 * (1 - norm.cdf(np.abs(weights_mean/weights_std))) # 2 tailed p value for the weight to have 0 mean
    
    
    significant_weights = np.where(p_values < 0.05) # finding the significant p values
    
    significant_weights = significant_weights[0]
    
    significant_weights_mean = weights_mean[significant_weights] # will be used in the plot
      
    
    plt.figure(figsize=(12, 8)) # plot of the weights and their 95% confidence intervals
    plt.grid() # grids so we can see the significant points are points which do not have a confidance interval which intercept 0 
    plt.errorbar(np.arange(0,100), weights_mean , yerr = _95_percent_confidence , ecolor='b', fmt='ok', capsize=3) # plotting all the weights
    plt.errorbar(significant_weights, significant_weights_mean, fmt='or') # marking the significant weights with red.
    plt.ylabel('Weight Values with 95% confidence intervals')
    plt.xlabel('Weight Indexes')
    plt.title('Ridge Regression for lambda = 0 with %95 confidence interval')
    
    print('I marked the significant weights with red')
    


def question1_part_c(): 
    
    print("\nAnswer of question 1 part c: \n")
     
    data = sio.loadmat('matlab_HW3_data2') # loading the data given to us
    Xn = data["Xn"]
    Yn = data["Yn"]
    
    opt_lamda = 353.5 # I wrote this value by hand here because its computation takes a very long time and re calculating it here would be excessive 
    
    weights = np.array(bootstrap(500, Xn, Yn, opt_lamda)) # getting the weights form 500 bootstrap iterations
    
    weights_mean = np.mean(weights, axis=0) # calculating the mean and standart deviation of each of the 100 weights
    weights_std = np.std(weights, axis=0)
    
     
    _95_percent_confidence = 2 * np.concatenate((weights_std.transpose(),weights_std.transpose()) , axis = 0) # the weights are normal distribution so 2 standart deviations to the both sides equals 95% confidence interval.
    
    
    p_values = 2 * (1 - norm.cdf(np.abs(weights_mean/weights_std))) # 2 tailed p value for the weight to have 0 mean
    
    
    significant_weights = np.where(p_values < 0.05) # finding the significant p values
    
    significant_weights = significant_weights[0]
    
    significant_weights_mean = weights_mean[significant_weights] # will be used in the plot
      
    print('I marked the significant weights with red')
    
    print('riged regresion has more significnat weights. Normaly we would expect the riged regresion to give out less number of significant parameters but here it also decresased the variance of the parameters which caused most of the weights which are on the edge of becoming significant signifiacant. Because of this riged regreasion version has more significant weights.')
    
    plt.figure(figsize=(12, 8)) # plot of the weights and their 95% confidence intervals
    plt.grid() # grids so we can see the significant points are points which do not have a confidance interval which intercept 0 
    plt.errorbar(np.arange(0,100), weights_mean , yerr = _95_percent_confidence , ecolor='b', fmt='ok', capsize=3) # plotting all the weights
    plt.errorbar(significant_weights, significant_weights_mean, fmt='or') # marking the significant weights with red.
    plt.ylabel('Weight Values with 95% confidence intervals')
    plt.xlabel('Weight Indexes')
    plt.title('Ridge Regression for lambda = ' + str(opt_lamda) + ' with %95 confidence interval')
    plt.show() # this privents the plot from closing themselves at the end 

    
    

    
def bootstrap_singel(iteration_count, x): # the function to callculate n bootstraps of a vector
    
    x_bootstraps = []
    
    for i in range(iteration_count):
        
        bootstrap_indexs = np.random.choice(np.arange(np.size(x)), np.size(x))  # chosing the index of the values which will be added to this iteration
        
        x_bootstrap = x[bootstrap_indexs] # adding the choosen values in to the iteration
        
        x_bootstraps.append(x_bootstrap) # adding the bootstrap iteration to the output list
        
    return np.array(x_bootstraps)    



def question2_part_a(): 
    
    print("\nAnswer of question 2 part a: \n")
    
    print("every time this code is run part a, part b, part c, part d and part e of this code will give a Slightly different output because of the randomness in bootstraping \n")
        
    data2 = sio.loadmat('matlab_HW3_data3') # loading the data given to us
    pop1 = data2["pop1"]
    pop2 = data2["pop2"]
    
    # null hypothesis is they are from the same distribution. so we will add the 2 sets and create bootstraps from that and then we will take the first 7 as the values in vox1 and the last 5 values as vox2. then we will compare their difrence in means and look at how probable it is to get the mean difrence we get in this particular case.
    
    pop1_pop2 = np.concatenate((pop1, pop2), axis = 1) # the neron group from our null hypotezis 
    
    bootstraps = bootstrap_singel(10000, pop1_pop2.transpose()) # creating 10000  bootstraps from the total population
    
    bootstrap_pop1 = bootstraps[:,0:7] # picking the first 7 elements as the pop1 for each of the 10000 bootstraps
    bootstrap_pop2 = bootstraps[:,7:12] # picking the last 5 elements as the pop2 for each of the 10000 bootstraps
    
    bootstrap_pop1_mean = np.mean(bootstrap_pop1, axis=1) # calculating mean of pop1 for each of the 10000 bootstraps
    bootstrap_pop2_mean = np.mean(bootstrap_pop2, axis=1) # calculating mean of pop2 for each of the 10000 bootstraps
    
    bootstrap_mean_difrence = bootstrap_pop1_mean - bootstrap_pop2_mean # calculating mean difrence for each of the 10000 bootstraps, this form the mean difrence distribution of the population.
    
    plt.figure() # I wanted to show it looks like a gausian
    plt.title('pop1, pop2 Mean Difference Distribution')
    plt.xlabel('E[pop1] - E[pop2]')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(bootstrap_mean_difrence, bins=29, density=True)
    
    # now we will compute the 2 tailed p value of getting the mean difrence for the origanal pop1, pop2 given to us. 
    
    given_mean_difrence = np.mean(pop1)-np.mean(pop2) # computing the mean difrence for the origanal pop1, pop2 given to us
    
    p = 2 * (1 - norm.cdf( ( np.abs(given_mean_difrence) - np.mean(bootstrap_mean_difrence)) / (np.std(bootstrap_mean_difrence)) )) # here we compute P(|bootstrap_mean_difrence| > |given_mean_difrence|) by using the CDF of the standart normla distribution.
    
    significant_points = np.where(np.abs(bootstrap_mean_difrence) > np.abs(given_mean_difrence)) # finding the significant points
    
    two_tailed_p_value = np.size(significant_points[0])/10000 # calcaulatin 2 tailed p value
    
    print("Two-tailed p-value for the null hypothesis that the two datasets follow the same distribution is " + str(two_tailed_p_value) + " for counting bootstrap results over the given difference.\nalso the p-value is " + str(p) + " for fitting a gausian to the null hypotehseies distiribuiton. \nThis is significantly lower than 0.05 so this hypothesis is incorect and the 2 populations follow different distributions.")



def bootstrap_double(iteration_count, x, y): # the function to callculate n bootstraps for 2 vectors with out breaking the coralation between them.
    
    x_bootstraps = []
    y_bootstraps = []
    
    
    for i in range(iteration_count):
        
        bootstrap_indexs = np.random.choice(np.arange(np.size(x)), np.size(x)) # chosing the index of the values which will be added to this iteration
        
        x_bootstrap = x[bootstrap_indexs] # adding the choosen values in to the iteration for x
        x_bootstraps.append(x_bootstrap) # adding the bootstrap iteration to the output list for x
           
        y_bootstrap = y[bootstrap_indexs] # adding the choosen values in to the iteration for y
        y_bootstraps.append(y_bootstrap) # adding the bootstrap iteration to the output list for y
           
    return np.array(x_bootstraps), np.array(y_bootstraps)



def question2_part_b(): 
    
    print("\nAnswer of question 2 part b: \n")
             
    data2 = sio.loadmat('matlab_HW3_data3') # loading the data given to us
    vox1 = data2["vox1"]
    vox2 = data2["vox2"]
    
    bootstraps_vox1, bootstraps_vox2 = bootstrap_double(10000, vox1.transpose(), vox2.transpose()) # creating 10000  bootstraps for vox1, vox2 pairs. they are row vectors and most fınctions are defiened for collum vector so we take their transpose.
    
    bootstraps_vox1_vox2_correlation = np.zeros(10000) # we will fill this in
    
    for i in range(10000): # Computing the correlation coaficent for 10000 bootstrap vox1, vox2 pairs.
        
        bootstraps_vox1_vox2_correlation[i] = (np.corrcoef((bootstraps_vox1[i,:]).transpose(), (bootstraps_vox2[i,:]).transpose() ))[0, 1] # comuting the correlation matrix for the ith bootstraps vox1 vox2 pair.
        
    plt.figure() # I wanted to show how the distribution looked
    plt.title('vox1, vox2 Correlation Coaficent Distribution')
    plt.xlabel('corr(vox1, vox2)')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(bootstraps_vox1_vox2_correlation, bins=29, density=True)
    
    bootstraps_vox1_vox2_correlation_mean = np.mean(bootstraps_vox1_vox2_correlation) # mean of the correlation
    
    # can use std to calculate because the distribution is not gausian.
    
    bootstraps_vox1_vox2_correlation_sorted = np.sort(bootstraps_vox1_vox2_correlation) # we order the correlation the pick the 250th and 9750th term because they should be on the 2 borders of 95% interval.
    
    lower_95_percent_confidence_border = bootstraps_vox1_vox2_correlation_sorted[249]
    upper_95_percent_confidence_border = bootstraps_vox1_vox2_correlation_sorted[9749]
    
    zero_correlation_points = np.where(bootstraps_vox1_vox2_correlation_sorted < 0.05) # the correlation values are floting point so if we where to look at exeactly 0 we woulden't get any matches
    
    percenteg_of_zero_points = np.size(zero_correlation_points[0]) * 100 / 10000 # turning the count to a percentage.
    
    print("Mean of the correlation coefficient distribution = " + str(bootstraps_vox1_vox2_correlation_mean))
    print("95% confidence interval of the correlation coefficient distribution = (" + str(lower_95_percent_confidence_border) + ", " + str(upper_95_percent_confidence_border) + ")")
    print("Percentile of bootstrap distribution, corresponding to a correlation value of 0 = " + str(percenteg_of_zero_points) + "%")
    


def question2_part_c(): 
    
    print("\nAnswer of question 2 part c: \n")
    
    data2 = sio.loadmat('matlab_HW3_data3') # loading the data given to us
    vox1 = data2["vox1"]
    vox2 = data2["vox2"]
    
    # null hypothesis is they have 0 correlation. so we will bootstrape the 2 sets seperatly this way if they have any corelation it will be broken, then we will look into its distribuiton and find the probability of getting the corelation etween the given vox1 and vox2    
    
    bootstraps_vox1 = bootstrap_singel(10000, vox1.transpose()) # creating 10000  bootstraps for vox1 and vox2 seperatly. they are row vectors and most fınctions are defiened for collum vector so we take their transpose.
    bootstraps_vox2 = bootstrap_singel(10000, vox2.transpose()) 
    
    bootstraps_vox1_vox2_correlation = np.zeros(10000) # we will fill this in
    
    for i in range(10000): # Computing the correlation coaficent for 10000 bootstrap vox1, vox2 pairs.
        
        bootstraps_vox1_vox2_correlation[i] = (np.corrcoef((bootstraps_vox1[i,:]).transpose(), (bootstraps_vox2[i,:]).transpose() ))[0, 1] # comuting the correlation matrix for the ith bootstraps vox1 vox2 pair.
        
    plt.figure() # I wanted to show how the distribution looked
    plt.title('vox1, vox2 Separately Boothstraped, Correlation Coaficent Distribution')
    plt.xlabel('corr(vox1, vox2)')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(bootstraps_vox1_vox2_correlation, bins=29, density=True)
    
    # now we will compute the 1 tailed p value of getting the correlation of the origanal vox1, vox2 given to us. 
        
    given_correlation = (np.corrcoef(vox1, vox2))[0, 1] # computing the correlation of the origanal vox1, vox2 given to us.
        
    p = 1 - norm.cdf(( (np.abs(given_correlation) - np.mean(bootstraps_vox1_vox2_correlation)) / (np.std(bootstraps_vox1_vox2_correlation)) )) # here we compute P(|bootstrap_mean_difrence| > |given_mean_difrence|) by using the CDF of the standart normla distribution.
         
    significant_points = np.where(bootstraps_vox1_vox2_correlation > given_correlation) # finding the significant points
    
    one_tailed_p_value = np.size(significant_points[0])/10000 # calculatin of 1 tailed p value
        
    print("One-tailed p-value for the null hypothesis that two voxel responses have zero correlation is " + str(one_tailed_p_value) + " for counting bootstrap results over the given correlation.\nalso the p-value is " + str(p) + " for fitting a gausian to the null hypotehseies distiribuiton. \nThis is significantly lower than 0.05 so this hypothesis is incorect and the 2 populations have positive correlaiton.\n")
    
    print("This makes sense because if we look at the result of part b the probability of getting a zero correlation was 0% and all the correlation coaficent we found were positive so the probability of these voxes having 0 or lower correlation being smaller then 5% is very logical.")
    
 

def bootstrap_same_polulation(iteration_count, x, y): # the function to callculate n bootstraps for 2 vectors asumming the 2 data came from the sane popolation an the responses have no difrence
    # I assume the first response for both data came from the same participant the second response for both data came from the same participant and so on.
    
    building_bootstraps = []
    face_bootstraps = []
    
    for i in range(iteration_count):
        
        bootstrap_indexs = np.random.choice(np.arange(np.size(x)), np.size(x)) # chosing the index of the values which will be added to this iteration, if we decide the ith response will be part of the boothstrap. Then it means we need to add the ith response of either one of the inputs to both of the outputs.
        
        building_bootstrap = []
        face_bootstrap = []
        
        
        for j in bootstrap_indexs: # for each participant we will decide which reponses to add the the 2 outputs
            
            random_bits = np.random.randint(2, size = 2) # we randomly choose which response will be added here
            
            if bool(random_bits[0]):
                
                building_bootstrap.append(x[j]) # adding the jth response of x with 50% probability
                
            else:
                
                building_bootstrap.append(y[j]) # adding the jth response of y with 50% probability
                
            if bool(random_bits[1]):
                
                face_bootstrap.append(x[j]) # adding the jth response of x with 50% probability
                
            else:
                
                face_bootstrap.append(y[j]) # adding the jth response of y with 50% probability
                 
        
        building_bootstraps.append(building_bootstrap) # adding the bootstrap iteration to the output list for building

        face_bootstraps.append(face_bootstrap) # adding the bootstrap iteration to the output list for y
           
    
    return np.array(building_bootstraps), np.array(face_bootstraps)  
 


def question2_part_d(): 
    
    print("\nAnswer of question 2 part d: \n")    
         
    data2 = sio.loadmat('matlab_HW3_data3') # loading the data given to us
    building = data2["building"]
    face = data2["face"]
          
    # null hypothesis is they have no difference in their response. If they have no difference in their response then their difrence in means will be zero. So we will compute the difference in means distribution asuming they have no difrence in their responses. then I will find the probability of obtaining the same difrence in means as the original given data.
    
    boothstrap_building, boothstrap_face = bootstrap_same_polulation(10000, building.transpose(), face.transpose())
    
    bootstrap_building_mean = np.mean(boothstrap_building, axis=1) # calculating mean of building for each of the 10000 bootstraps
    bootstrap_face_mean = np.mean(boothstrap_face, axis=1) # calculating mean of face for each of the 10000 bootstraps
    
    bootstrap_mean_difrence = bootstrap_building_mean - bootstrap_face_mean # calculating mean difrence for each of the 10000 bootstraps, this form the mean difrence distribution of the population.
    
    plt.figure() # I wanted to show how it looks
    plt.title('Building, Face Mean Difference Distribution \nFor the Same Participants')
    plt.xlabel('E[pop1] - E[pop2]')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(bootstrap_mean_difrence, bins=29, density=True)
       
    given_mean_difrence = np.mean(building)-np.mean(face) # computing the mean difrence for the origanal pop1, pop2 given to us
    
    p = 2 * (1 - norm.cdf(( ( np.abs(given_mean_difrence) - np.mean(bootstrap_mean_difrence)) / (np.std(bootstrap_mean_difrence)) ))) # here we compute P(|bootstrap_mean_difrence| > |given_mean_difrence|) by using the CDF of the standart normla distribution.
    
        
    significant_points = np.where(np.abs(bootstrap_mean_difrence) > np.abs(given_mean_difrence)) # finding the significant points

    
    two_tailed_p_value = np.size(significant_points[0])/10000 # calculatin of 2 tailed p value
    
    
    print("Two-tailed p-value for the null hypothesis that building and face create the same response is " + str(two_tailed_p_value) + " for counting bootstrap results over the given difference. \nalso the p-value is " + str(p) + " for fitting a gausian to the null hypotehseies distiribuiton. \nThis is significantly lower than 0.05 so this hypothesis is incorect and buildings and faces create different responses.")


def question2_part_e(): 
    
    print("\nAnswer of question 2 part e: \n")
           
    data2 = sio.loadmat('matlab_HW3_data3') # loading the data given to us
    building = data2["building"]
    face = data2["face"]
    
    # null hypothesis is they have no difference in their response. If they have no difference in their response then their difrence in means will be zero. So we will compute the difference in means distribution asuming they have no difrence in their responses. then I will find the probability of obtaining the same difrence in means as the original given data.
    # what we do in this question is same as part a.    
    
    building_face = np.concatenate((building, face), axis = 1) # the neron group from our null hypotezis 
       
    bootstraps = bootstrap_singel(10000, building_face.transpose()) # creating 10000  bootstraps from the total population
       
    bootstrap_building = bootstraps[:,0:20] # picking the first 20 elements as the building for each of the 10000 bootstraps
    bootstrap_face = bootstraps[:,20:40] # picking the last 20 elements as the face for each of the 10000 bootstraps
       
    bootstrap_building_mean = np.mean(bootstrap_building, axis=1) # calculating mean of building experiment for each of the 10000 bootstraps
    bootstrap_face_mean = np.mean(bootstrap_face, axis=1) # calculating mean of face experiment for each of the 10000 bootstraps
       
    bootstrap_mean_difference = bootstrap_building_mean - bootstrap_face_mean # calculating mean difference for each of the 10000 bootstraps, this form the mean difference distribution of the population.
      
  
    given_mean_difrence = np.mean(building)-np.mean(face) # computing the mean difrence for the origanal building, face given to us
    
    p = 2 * (1 - norm.cdf(( ( np.abs(given_mean_difrence) - np.mean(bootstrap_mean_difference) ) / (np.std(bootstrap_mean_difference)) ))) # here we compute P(|bootstrap_mean_difrence| > |given_mean_difrence|) by using the CDF of the standart normla distribution.
    
            
    significant_points = np.where(np.abs(bootstrap_mean_difference) > np.abs(given_mean_difrence)) # finding the significant points
    
    two_tailed_p_value = np.size(significant_points[0])/10000 # calculatin of 2 tailed p value

    print("Two-tailed p-value for the null hypothesis that building and face create the same response is " + str(two_tailed_p_value) + " for counting bootstrap results over the given difference. \nalso the p-value is " + str(p) + " for fitting a gausian to the null hypotehseies distiribuiton. \nThis is significantly lower than 0.05 so this hypothesis is incorect and buildings and faces create different responses.")
    print("\nThis means there is a different response for building and face, regardless of our assumption of the participitants.")

    plt.figure() # I wanted to show how it looks
    plt.title('Building, Face Response Mean Difference Distribution \nFor Different Participants')
    plt.xlabel('E[pop1] - E[pop2]')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.hist(bootstrap_mean_difference, bins=29, density=True)
    plt.show() # this privents the plot from closing themselves at the end     


Batu_Arda_Düzgün_21802633_hw3(question) # this is the only line which runs so it is very important.
    
    
    
    
    
    
    
    
    
    
