
import sys # the primer already has it
import numpy as np # this brings simple matrix operations to python
import matplotlib.pyplot as plt # this brings nice loking plots to python

import random # for random number generation in some verifications

from scipy import linalg as lg # this brings complicated matrix operations to python and elivates it to the level of matlab

import math as math # this brings mathematical operations to python


"""
If you want to run this code in a IDE you need to commend out the line below and write the line
question = '1'
then run the code for question 1 and write the line 
question = '2'
then run the code for question 2 
"""
question = sys.argv[1]

def Batu_Arda_Düzgün_21802633_hw1(question):
    if question == '1' :
        print("Answer of question 1.) \n")
        question1_part_a()        
        question1_part_b()
        question1_part_c()        
        question1_part_d()
        question1_part_e()        
        question1_part_f()
    elif question == '2' :
        print("Answer of question 2.) \n")
        question2_part_a()
        question2_part_b()
        question2_part_c()   
        question2_part_d()   
        question2_part_e()


def question1_part_a():
    
    print("\nAnswer of question 1 part a: \n")
    
    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question

    NS = lg.null_space(A) # this finds the null space of the given matrix
        

    
    CNS = NS.copy() # I am doing these steps to put the null space in to a format close to what I found by hand.
    
    C0 = CNS[-1,0]
    C1 = CNS[-1,1]
    
    CNS[: ,0] -= CNS[: , 1] * C0 / C1

    C0 = CNS[-2,0]   
    
    CNS[: ,0] /= C0
    
    C0 = CNS[-2,0]
    C1 = CNS[-2,1]
     
    CNS[: ,1] -= CNS[: , 0] * C1 / C0    
    
    C1 = CNS[-1,1]   
    
    CNS[: ,1] /= C1
    
    print("The null space of the matrix is ")
    print(CNS)

 
    
def question1_part_b():
    
    print("\nAnswer of question 1 part b: \n")
     
    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question
    
    x_p = np.array([ [1] , [2] , [0] , [0] ]) # this is the instensiation of the x_p I calculated by hand
    
    b = A.dot(x_p)
    
    print("For the particular solution I found " + str(x_p.transpose()) + "^T the result is " + str(b.transpose()) + "^T which means it satisfys the equation")
    
    
def question1_part_c():
    
    print("\nAnswer of question 1 part c: \n")

    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question
    
    x_p = np.array([ [1] , [2] , [0] , [0] ]) # this is the instensiation of the x_p I calculated by hand

    x3 = random.random() # this generates a random number between 0 and 1. we can multipley 1000 to get a number generation between 0 and 1000 but I didn't do it because it shoudln't change the realizim of the test

    x4 = random.random()
    
    x3x4 = np.array([ [x3] , [x4] ])

    null_space = lg.null_space(A) # this finds the null space of the given matrix

    x = x_p + null_space.dot(x3x4) # creating the random x from our solution set.
    
    b = A.dot(x)

    print( "for x_3 = " + str(x3) + " and x_4 = " + str(x4) + " b is equal to: ")
    print( b )


def question1_part_d():
 
    print("\nAnswer of question 1 part d: \n")
    
    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question   
    
    pseudo_A = np.linalg.pinv(A) # this is the function to compute the pseudo inverse of a matrix
    
    print("Pseudo invers of the matrix A is: ")
    print(pseudo_A)
    
    
    
def question1_part_e():
    
    print("\nAnswer of question 1 part e: \n")
    
    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question 
    
    x_sparsest_1 = np.array([ [1] , [2] , [0] , [0] ]) # I calculated this by hand
    
    x_sparsest_2 = np.array([ [0] , [3] , [-1] , [0] ]) # I calculated this by hand
    
    x_sparsest_3 = np.array([ [3] , [0] , [2] , [0] ]) # I calculated this by hand
    
    x_sparsest_4 = np.array([ [0] , [1.5] , [0] , [0.5] ]) # I calculated this by hand
    
    x_sparsest_5 = np.array([ [-3] , [0] , [0] , [2] ]) # I calculated this by hand
    
    x_sparsest_6 = np.array([ [0] , [0] , [1] , [1] ]) # I calculated this by hand
    
    b1 = A.dot(x_sparsest_1)
    b2 = A.dot(x_sparsest_2)
    b3 = A.dot(x_sparsest_3)
    b4 = A.dot(x_sparsest_4)
    b5 = A.dot(x_sparsest_5)
    b6 = A.dot(x_sparsest_6)
    
    print(str(x_sparsest_1.transpose()) + "^T as a x vector gives the output of " + str(b1.transpose()) + "^T \n")
    print(str(x_sparsest_2.transpose()) + "^T as a x vector gives the output of " + str(b2.transpose()) + "^T \n")
    print(str(x_sparsest_3.transpose()) + "^T as a x vector gives the output of " + str(b3.transpose()) + "^T \n")
    print(str(x_sparsest_4.transpose()) + "^T as a x vector gives the output of " + str(b4.transpose()) + "^T \n")
    print(str(x_sparsest_5.transpose()) + "^T as a x vector gives the output of " + str(b5.transpose()) + "^T \n")
    print(str(x_sparsest_6.transpose()) + "^T as a x vector gives the output of " + str(b6.transpose()) + "^T \n")
    
def question1_part_f(): 

    print("\nAnswer of question 1 part f: \n")       
    
    A = np.array([ [1, 0, -1, 2] , [2, 1, -1, 5] , [3, 3, 0, 9]]) # this is the instensiation of the A matrix given in the question   
    b = np.array([ [1] , [4] , [9] ]) # this is the instensiation of the b vector given in the question 

    S = lg.lstsq(A, b) # this is the function to compute the least norm solution of Ax = b

    print("least-norm solution to Ax = b is")
    print(S[0]) # lstsq returns a list of answers but we only want the first term of that lis which is the least norm solution.
    
def question2_part_a():
    
    print("\nAnswer of question 2 part a: \n")
    
    print("All the plots will be displayed at the end")
    
    Xaxis = np.arange(0, 1.001, 0.001) # python doesn't take 1.001 because it is the last elemnent so I had to stop at 1.001 to take 1 as the last element
    
    Prob_data_language_given_x = Bernuoli_distribution(869, 103, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    Prob_data_nonlanguage_given_x = Bernuoli_distribution(2353, 199, Xaxis) # this places each value of the array in the funtion and creats an array out of its results

    plt.figure()
    plt.bar(Xaxis, Prob_data_language_given_x, align = "edge", width = 0.001)    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('likelihood of observed frequencies')    
    plt.title('likelihood of observed activation frequencies vs bernoulli \n probability parameters for tasks involving language')
    plt.axis([0, 0.2, 0, 0.05])
        
    plt.figure()
    plt.bar(Xaxis, Prob_data_nonlanguage_given_x, align = "edge", width = 0.001, facecolor='g')    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('likelihood of observed frequencies')    
    plt.title('likelihood of observed activation frequencies vs bernoulli \n probability parameters for tasks not involving language')
    plt.axis([0, 0.2, 0, 0.05])
    
    
def Bernuoli_distribution(number_of_trys , number_of_ones , p):

    return ( math.factorial(number_of_trys)/(math.factorial(number_of_ones) * math.factorial(number_of_trys - number_of_ones)) * (p ** number_of_ones) * ((1 - p) ** (number_of_trys - number_of_ones)) ) # this is the formula of a binomial distrubution
    
    
def question2_part_b(): 
    
    print("\nAnswer of question 2 part b: \n")
        
    Xaxis = np.arange(0, 1.001, 0.001) # python doesn't take 1.001 because it is the last elemnent so I had to stop at 1.001 to take 1 as the last element
    
    Prob_data_language_given_x = Bernuoli_distribution(869, 103, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    Prob_data_nonlanguage_given_x = Bernuoli_distribution(2353, 199, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    
    print("maximum likelihood of x_l = " + str( round( max(Prob_data_language_given_x) , 4 )) + " and the Bernoulli paremeter that achives this is " + str(np.argmax(Prob_data_language_given_x) / 1000))
    print("")
    print("maximum likelihood of x_nl = " + str( round( max(Prob_data_nonlanguage_given_x) , 4 )) + " and the Bernoulli paremeter that achives this is " + str(np.argmax(Prob_data_nonlanguage_given_x) / 1000))
    
    


def question2_part_c(): 
    
    print("\nAnswer of question 2 part c: \n")
    
    Xaxis = np.arange(0, 1.001, 0.001) # python doesn't take 1.001 because it is the last elemnent so I had to stop at 1.001 to take 1 as the last element
    
    Prob_data_language_given_x = Bernuoli_distribution(869, 103, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    Prob_data_nonlanguage_given_x = Bernuoli_distribution(2353, 199, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    prob_x = 1 / 1001
    
    Prob_data_language = sum(Prob_data_language_given_x) * prob_x 
    
    Prob_data_nonlanguage = sum(Prob_data_nonlanguage_given_x) * prob_x
    
    
    X_given_data_language = prob_x * Prob_data_language_given_x / Prob_data_language # bayes' rule formula
    
    X_given_data_nonlanguage = prob_x * Prob_data_nonlanguage_given_x / Prob_data_nonlanguage # bayes' rule formula
    
    plt.figure()
    plt.bar(Xaxis, X_given_data_language, align = "edge", width = 0.001)    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('P(X | data) posterior distribution')    
    plt.title('posterior distributions P(X|data) vs bernoulli probability \n parameters for tasks involving language')
    plt.axis([0, 0.2, 0, 0.08])
    
    plt.figure()
    plt.bar(Xaxis, X_given_data_nonlanguage, align = "edge", width = 0.001, facecolor='g')    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('P(X | data) posterior distribution')   
    plt.title('posterior distributions P(X|data) vs bernoulli probability \n parameters for tasks not involving language')
    plt.axis([0, 0.2, 0, 0.08])
    
    
    cumulative_X_given_data_language = np.cumsum(X_given_data_language)    
    cumulative_X_given_data_nonlanguage = np.cumsum(X_given_data_nonlanguage)
    
    plt.figure()
    plt.bar(Xaxis, cumulative_X_given_data_language, align = "edge", width = 0.001)    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('P(X | data) cumulative distribution')    
    plt.title('cumulative distributions P(X|data) vs bernoulli probability \n parameters for tasks involving language')
    plt.axis([0, 0.2, 0, 1.1])
    
    plt.figure()
    plt.bar(Xaxis, cumulative_X_given_data_nonlanguage, align = "edge", width = 0.001 , facecolor='g')    
    plt.xlabel('bernoulli probability parameters')
    plt.ylabel('P(X | data) cumulative distribution')   
    plt.title('cumulative distributions P(X|data) vs bernoulli probability \n parameters for tasks not involving language')
    plt.axis([0, 0.2, 0, 1.1])
    
    for i in range(0 , 1002): # we will loop trough all the values of the array
        
        if cumulative_X_given_data_language[i] >= 0.025: # break out of the loop once we find the lower confidence bound
            
            x_language_lower_95_confidence = Xaxis[i]
            
            break

    for i in range(0 , 1002): # we will loop trough all the values of the array
        
        if cumulative_X_given_data_language[i] >= 0.975: # break out of the loop once we find the upper confidence bound
            
            x_language_upper_95_confidence = Xaxis[i]
            
            break
        
        
    for i in range(0 , 1002): # we will loop trough all the values of the array
        
        if cumulative_X_given_data_nonlanguage[i] >= 0.025: # break out of the loop once we find the lower confidence bound
            
            x_non_language_lower_95_confidence = Xaxis[i]
            
            break
        
        
    for i in range(0 , 1002): # we will loop trough all the values of the array
        
        if cumulative_X_given_data_nonlanguage[i] >= 0.975: # break out of the loop once we find the upper confidence bound
            
            x_non_language_upper_95_confidence = Xaxis[i]
            
            break

    
    print("lower 95% confidence bound of x_language: " + str(x_language_lower_95_confidence))
    print("upper 95% confidence bound of x_language: " + str(round(x_language_upper_95_confidence , 3)))
    print("")
    print("lower 95% confidence bound of x_nonlanguage: " + str(x_non_language_lower_95_confidence))
    print("upper 95% confidence bound of x_nonlanguage: " + str(x_non_language_upper_95_confidence))
    
    
def question2_part_d():   
    
    print("\nAnswer of question 2 part d: \n")
    
    Xaxis = np.arange(0, 1.001, 0.001) # python doesn't take 1.001 because it is the last elemnent so I had to stop at 1.001 to take 1 as the last element
    
    Prob_data_language_given_x = Bernuoli_distribution(869, 103, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    Prob_data_nonlanguage_given_x = Bernuoli_distribution(2353, 199, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    prob_x = 1 / 1001
    
    Prob_data_language = sum(Prob_data_language_given_x) * prob_x 
    
    Prob_data_nonlanguage = sum(Prob_data_nonlanguage_given_x) * prob_x
    
    
    X_given_data_language = prob_x * Prob_data_language_given_x / Prob_data_language # bayes' rule formula
    
    X_given_data_nonlanguage = prob_x * Prob_data_nonlanguage_given_x / Prob_data_nonlanguage # bayes' rule formula
    
    X_l_X_nl = np.outer(X_given_data_nonlanguage, X_given_data_language)
    
    plt.figure()
    plt.imshow(X_l_X_nl , origin='lower' , extent=[0,1,0,1]) #this is the function to plot a heat map
    plt.colorbar()
    plt.xlabel("x_l")
    plt.ylabel("x_nl")
    plt.title("Joint Posterior Distribution\n P(X_l, X_nl | data)")
    
    plt.figure()
    plt.imshow(X_l_X_nl , origin='lower' , extent=[0,1,0,1]) #this is the function to plot a heat map
    plt.axis([0, 0.2, 0, 0.2])
    plt.colorbar()
    plt.xlabel("x_l")
    plt.ylabel("x_nl")
    plt.title("Joint Posterior Distribution zoomed in\n P(X_l, X_nl | data)")
 
    xl_bigger_xnl = 0    
    xnl_bigger_xl = 0  


    for i in range(0 , 1001): # this loops trough all the matrix and put its upper and lower triangles entries to the respective sums.
        for j in range(0 , 1001):
            if j > i :
                xl_bigger_xnl += X_l_X_nl[i , j]
            else:
                xnl_bigger_xl += X_l_X_nl[i , j]
                
                
    print("P(Xl > Xnl|data) = " + str(round(xl_bigger_xnl , 4)))
    print("P(Xl ≤ Xnl|data) = " + str(round(xnl_bigger_xl , 4)))         
        
def question2_part_e():

    print("\nAnswer of question 2 part e: \n")    

    Xaxis = np.arange(0, 1.001, 0.001) # python doesn't take 1.001 because it is the last elemnent so I had to stop at 1.001 to take 1 as the last element
    
    Prob_data_language_given_x = Bernuoli_distribution(869, 103, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    Prob_data_nonlanguage_given_x = Bernuoli_distribution(2353, 199, Xaxis) # this places each value of the array in the funtion and creats an array out of its results
    
    P_activation_given_language = max(Prob_data_language_given_x)
    P_activation_given_nonlanguage = max(Prob_data_nonlanguage_given_x)

    P_language = 0.5 # this is given the question.

    P_activation = P_language * P_activation_given_language + ( 1 - P_language ) * P_activation_given_nonlanguage

    P_language_given_activation = P_activation_given_language * P_language / P_activation # Bayes' rule formula

    print ("P(language|activation) = " + str(round(P_language_given_activation , 4)))

    plt.show() # this privents the plot from closing themselfs at the end
 
Batu_Arda_Düzgün_21802633_hw1(question)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
