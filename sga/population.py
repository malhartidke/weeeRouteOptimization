import numpy as np

###################### Formation of Individual for SGA ##################################
#                                                                                       #
#  No Input Parameters                                                                  #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  param: Inidividual Formed                                                            #
#                                                                                       #
#########################################################################################
def give_individual():
    
    param = np.zeros(9)
    
    param[0] = np.random.random()                
    param[1] = np.random.random()
    param[2] = np.random.random()
    param[3] = np.random.random()
    param[4] = np.random.random()
    param[5] = np.random.random()
    param[6] = np.random.random()
    param[7] = np.random.randint(2,3,1)
    param[8] = np.random.randint(2,3,1)
    
    return(param)
#---------------------------------------------------------------------------------------#