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
from acs.cvrp import run_aco
from deap import base, creator, tools
from contextlib import contextmanager
#from Generate_Data import gen_data
#from Distance_calc import dist_mat


# To print coefficients array in one line
@contextmanager
def print_array_on_one_line():
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf)
    yield
    np.set_printoptions(**oldoptions)

def main(toolbox):
    
    print('\nLoading Data Files')
    locCount = 0

    # Variable storing the location of files with file name
    file_locationLoc = 'depot_location.csv'
    file_locationVehicle = 'vehicle_info.csv'
    file_location = 'time_loc_data.csv'
    file_locationM = 'Dist_Matrix.csv'
    file_locationApp = 'wasteVolumeMultipliers.csv'
    file_locationTime = 'BreakTimes.csv'
    file_locationParameters = 'parameters.csv'                                                

    try:
        
        # Reading the CSV files
        dfDepot = pd.read_csv(file_locationLoc)
        dfVehicle = pd.read_csv(file_locationVehicle)
        df = pd.read_csv(file_location)
        dfM = pd.read_csv(file_locationM)
        dfApp = pd.read_csv(file_locationApp)
        dfTime = pd.read_csv(file_locationTime)
        dfParam = pd.read_csv(file_locationParameters)

    # Error Handling
    except FileNotFoundError:                                                     
        
        print('\nThere was problem in loading files. Please check the name of files.')   
        sys.exit()

    print('Files loaded succesfully')

    depot_loc_data = np.array(dfDepot)
    max_num_vehicles = dfVehicle['Max Number of Vehicles'][0]
    capacity = dfVehicle['Capacity of Vehicles'][0]
    best = np.zeros((np.size(depot_loc_data,0),11),dtype=object)
    best_fitness_value = np.zeros(np.size(depot_loc_data,0),dtype=object)
    
    
    for lat, lon in depot_loc_data:

        raw_call_data = np.array(df)
        dis_mat = np.array(dfM)
        wasteValues = np.asarray(dfApp)[:,2]
        wasteEnvValues = np.asarray(dfApp)[:,3]
        breakTimes = np.asarray(dfTime)[:,1]
        
        # Creating and appending value and environmental of Loads
        wasteValues = np.reshape(wasteValues,(wasteValues.size,-1))
        wasteEnvValues = np.reshape(wasteEnvValues,(wasteEnvValues.size,-1))
        loadValue = np.matmul(raw_call_data[:,7:12], wasteValues)
        envValue = np.matmul(raw_call_data[:,7:12], wasteEnvValues)
        call_data = np.append(raw_call_data[:,0:7],loadValue,axis=1)
        call_data = np.append(call_data,envValue,axis=1)

        # np.savetxt('tim_loc_dataValues.csv',call_data,delimiter=",")

        # Calculating Break Start and End Times and Service End Time
        breakTimeStart = (breakTimes[2] - breakTimes[1])*60
        breakTimeEnd = breakTimeStart + (breakTimes[3]*60)
        endTime = breakTimes[0]*60

        # Calculating Travel Time Matrix
        travel_time_mat = dis_mat/333.3333
        
        # Run the SGA Algorithm
        best[locCount,:], best_fitness_value[locCount] = run_sga(call_data, dis_mat, travel_time_mat, lat, lon, capacity, max_num_vehicles, breakTimeStart, breakTimeEnd, endTime, locCount, dfParam, toolbox)
        locCount += 1

    bestIdx = np.argmax(best_fitness_value)
    bestOfBest = best[bestIdx,:]
    bestOfBest_fitness_value = best_fitness_value[bestIdx]

    with print_array_on_one_line():
        print("\nBest individual is %s, %s" % (bestOfBest, bestOfBest_fitness_value))
    print("\nThe most preferred location would be: Latitude= %s  Longitude=%s" % (depot_loc_data[bestIdx,0],depot_loc_data[bestIdx,1]))
    fitnessValue = run_aco(call_data, dis_mat, travel_time_mat, lat, lon, capacity, max_num_vehicles, breakTimeStart, breakTimeEnd, endTime, bestOfBest[:9],np.absolute(bestOfBest[9:11].astype(int)),finalPrint=1)

    #Generating CSV Files
    print('\nGenerating CSV files..........')

    try:
        
        # File containing the best obtained values of parameters
        np.savetxt("best_coef.csv", bestOfBest, delimiter=",")       

    except PermissionError:
        
        print("The file is being used in another program or by another user. Please close the file and run this program again")
        print('CSV file could not be generated..........')
        print('The program has terminated')    
        sys.exit()

    print('CSV files generated succesfully...........')

    # Calculate the time required to run the whole program
    time_elapsed = (timer() - start_time)
    print('\nTime taken to run the code:',time_elapsed)

file_locationParameters = 'parameters.csv'
dfParam = pd.read_csv(file_locationParameters)

# Creating a class 'FitnessMax' based on class 'base.Fitness'
# with the objective of maximizing single fitness value which
# is denoted by positive single value of weight
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

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
toolbox.register("mate", SimulatedBinary,q=dfParam['Mating Parameter q'][0])

# Assigning the mutation function and providing the crossover parameters                                                
toolbox.register("mutate", mutPolynomial,delta_max=dfParam['Crossover Parameter delta_max'][0],q=dfParam['Crossover Parameter q'][0])                                  

# Assigning the selection function and providing the tournament selection parameters
toolbox.register("select", tools.selTournament, tournsize=dfParam['SGA Tournament Size'][0].astype(int))


if __name__ == '__main__':
    main(toolbox)