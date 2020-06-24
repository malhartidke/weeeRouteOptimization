import numpy as np

####################### Implmenting Crossover Algorithm ################################
# This function performs mutation on a set of parameters recieved                      #
# Name of Method: Polynomial Mutation                                                  #
# This function is based on mathematical implementation                                #
#                                                                                      #
# Description of Input Parameters:                                                     #
# in_mutant:      Set of parameters                                                    #
# delta_max:      Maximum Pertubration Factor                                          #
# q:              Exponent for Crossover                                               #
#                                                                                      #
# Description of Output Parameter:                                                     #
# in_mutant:      Set of Parameters after Mutation                                     #
#                                                                                      #
########################################################################################
def mutPolynomial(in_mutant,delta_max,q):

    for idx in range(in_mutant.size):
        
        rand_no = np.random.random()
        
        if (rand_no < 0.5):
        
            delta = pow((2*rand_no),(1/(q+1))) - 1
        
        elif (rand_no > 0.5):
        
            delta = 1 - pow((2*(1-rand_no)),(1/(q+1)))
        
        else:
        
            delta = 1
        
        in_mutant[idx] += (delta*delta_max)
    
    return in_mutant,
#---------------------------------------------------------------------------------------#