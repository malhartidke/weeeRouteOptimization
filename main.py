# Importing the required libraries
from timeit import default_timer as timer
start_time = timer()
import sys
import numpy as np
import pandas as pd
from sga.ga_aco import run_sga
from sga.population import give_individual
from sga.crossover import SimulatedBinary
from sga.mutation import mutPolynomial
from deap import base, creator, tools
#from Generate_Data import gen_data
#from Distance_calc import dist_mat

def main(toolbox):
    

    lat = 26.9921975
    lon = 75.7840554

    # Variable storing the location of file with file name
    file_location = 'time_loc_data.csv'
    file_locationM = 'Dist_Matrix.csv'                                                

    try:
        
        # Reading the CSV file
        df = pd.read_csv(file_location)
        dfM = pd.read_csv(file_locationM)

    # Error Handling
    except FileNotFoundError:                                                     
        
        sys.exit()

    call_data = np.array(df)
    dis_mat = np.array(dfM)

    # Calculating Travel Time Matrix
    travel_time_mat = dis_mat/20
        
    # Run the SGA Algorithm
    best, best_fitness_value = run_sga(call_data, dis_mat, travel_time_mat, lat, lon, toolbox)

    print("\nBest individual is %s, %s" % (best, best_fitness_value))

    #Generating CSV Files
    print('\nGenerating CSV files..........')

    try:
        
        # File containing the best obtained values of parameters
        np.savetxt("best_coef.csv", best, delimiter=",")       

    except PermissionError:
        
        print("The file is being used in another program or by another user. Please close the file and run this program again")
        print('CSV file could not be generated..........')
        print('The program has terminated')    
        sys.exit()

    print('CSV files generated succesfully...........')

    # Calculate the time required to run the whole program
    time_elapsed = (timer() - start_time)
    print('\nTime taken to run the code:',time_elapsed)

# Creating a class 'FitnessMax' based on class 'base.Fitness'
# with the objective of maximizing single fitness value which
# is denoted by positive single value of weight
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))

# Creating a class 'Individual' as a numpy array
# whose fitness is defined by the 'FitnessMax' class
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Renaming the function 'give_individual' to 'one_set_para' which returns a set of parameter when called
toolbox.register("one_set_para",give_individual)

# Individual is created by calling 'one_set_para' once                                             
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.one_set_para) 

# Population is created by calling individual 'n' times; Here 'n' is specified when the population function is called
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                                 

# Assigning the mating function and providing the mating parameters
toolbox.register("mate", SimulatedBinary,q=5)

# Assigning the mutation function and providing the crossover parameters                                                
toolbox.register("mutate", mutPolynomial,delta_max=0.1,q=5)                                  

# Assigning the selection function and providing the tournament selection parameters
toolbox.register("select", tools.selTournament, tournsize=2)


if __name__ == '__main__':
    main(toolbox)