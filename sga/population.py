import numpy as np
import pandas as pd

###################### Formation of Individual for SGA ##################################
#                                                                                       #
#  No Input Parameters                                                                  #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  param: Inidividual Formed                                                            #
#                                                                                       #
#########################################################################################
def give_individual():
    
    param = np.zeros(11)
    file_locationParameters = 'parameters.csv'
    dfParam = pd.read_csv(file_locationParameters)

    param[0] = np.random.random()                
    param[1] = np.random.random()
    param[2] = np.random.random()
    param[3] = np.random.random()
    param[4] = np.random.random()
    param[5] = np.random.random()
    param[6] = np.random.random()
    param[7] = np.random.random()
    param[8] = np.random.random()
    param[9] = np.random.randint(2,dfParam['ACO Population Limit'][0].astype(int),1)
    param[10] = np.random.randint(2,dfParam['ACO Max No, of Iterations'][0].astype(int),1)
    
    return(param)
#---------------------------------------------------------------------------------------#