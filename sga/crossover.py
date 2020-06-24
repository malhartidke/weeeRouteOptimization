import numpy as np

####################### Implmenting Crossover Algorithm ################################
# This function performs crossover between two set of parameters recieved              #
# Name of Cross-over Method: Simulated Binary Crossover                                #
# This function is based on mathematical implementation                                #
# Reference of mathematical implementation: "Simulated Binary Crossover for Continous  # 
# Search Space" by K. Deb, R. Agrawal in Journal named "Complex Systems" in June 2000  #
#                                                                                      #
# Description of Input Parameters:                                                     #
# ind1, ind2:      Sets of parameters                                                  #
# q:               Exponent for Crossover                                              #
#                                                                                      #
# Description of Output Parameter:                                                     #
# ind1, ind2:      Set of Parameters after Crossover                                   #
#                                                                                      #
########################################################################################
def SimulatedBinary(ind1, ind2,q):

    for idx, Pr1, Pr2 in enumerate(zip(ind1, ind2)):
        
        rand_no = np.random.random()
        
        if (rand_no < 0.5):
            
            alpha_prime = pow((2*rand_no),(1/(q+1)))
        
        elif (rand_no > 0.5):
            
            alpha_prime = pow((1/(2*(1-rand_no))),(1/(q+1)))
        
        else:
            
            alpha_prime = 1
        
        t1 = 0.5*(Pr1+Pr2)
        t2 = 0.5*alpha_prime*abs((Pr1-Pr2))

        ind1[idx] = t1 - t2
        ind2[idx] = t1 + t2

    return ind1, ind2
#---------------------------------------------------------------------------------------#