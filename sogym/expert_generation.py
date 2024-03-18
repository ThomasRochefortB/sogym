
from sogym.env import sogym
from sogym.mmc_optim import run_mmc
import datetime
import os
import json
import numpy as np
import codecs
import multiprocessing as mp
import glob
from sogym.struct import build_design
from imitation.data.types import Trajectory
from itertools import permutations
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import glob
import re
from multiprocessing import Pool, cpu_count
import pickle

# Let's load an expert sample:
import json
def count_top (filepath):
    #function to count the number of .json in the /dataset folder:
    path, dirs, files = next(os.walk(filepath))
    file_count = len(files)
    return file_count



def load_top (filepath):
    #function to load the .json in the /dataset folder:
    obj_text = codecs.open(filepath, 'r', encoding='utf-8').read()
    dict = json.loads(obj_text)
    
    return dict

        
def generate_mmc_solutions(key,dataset_folder):
    """
    This function generates a single solution for the mmc problem and saves it in the dataset folder

    Args:
        key (int): The key for the multiprocessing

    Returns:
        None: The function saves the solution in the dataset folder

    """
    env = sogym(mode='train',observation_type='dense',vol_constraint_type='soft',seed=key)
    obs = env.reset()
    xval, f0val, num_iter, H, Phimax, allPhi, den,N=  run_mmc(env.conditions,env.nelx,env.nely,env.dx,env.dy,plotting='nothing',verbose=0)
    #out_conditions = train_env.conditions.copy()
    out_conditions = env.conditions.copy()

    # train_env.conditions contains numpy arrays that I cant save in the json file. I need to convert them to lists
    for dict_key in out_conditions.keys():
        if type(out_conditions[dict_key]) is np.ndarray:
            out_conditions[dict_key] = out_conditions[dict_key].tolist()
            # convert also the internals of the key in case I have arrays of arrays:
        if  type(out_conditions[dict_key]) is list and type(out_conditions[dict_key][0]) is np.ndarray:
            out_conditions[dict_key] = [float(x[0])for x in out_conditions[dict_key]]
        # Convert all of the int64 to floats:
    # Generate a dictionary to save the json with the following keys:
    
    save_dict={
        'boundary_conditions':out_conditions,
        'number_components':N,
        'compliance':f0val.tolist(),
        'num_iter':float(num_iter),
        'design_variables':xval.tolist(),
        'dx': env.dx,
        'dy': env.dy,
        'nelx': env.nelx,
        'nely': env.nely,
        'h': H.tolist(),
        'phimax':Phimax.tolist(),
        'phi':allPhi.tolist(),
        'den':den.tolist(),
        'extra': "Optimal topologies, maxiter=1000"
    }

    # Generate a timestamp with miliseconds for the filename:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        # open a file for writing
    with open('{}/{}'.format(dataset_folder,timestamp) +'.json', 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump(save_dict, f)

def generate_dataset(dataset_folder, num_threads=1, num_samples=10):
    '''
    This function generates a dataset of solutions for the mmc problem and saves it in the dataset folder

    Args:
        num_threads (int, optional): Number of threads to use for multiprocessing. Defaults to 1.
        num_samples (int, optional): Number of samples to generate. Defaults to 10.
    '''
    if num_threads >1:
        nbrAvailCores=num_threads
        pool = mp.Pool(processes=nbrAvailCores)
        resultsHandle = [pool.apply_async(generate_mmc_solutions, args=(z,dataset_folder)) for z in range(0,num_samples)]
        results = [r.get() for r in tqdm(resultsHandle)]
        pool.close()
    else:

        for i in tqdm.tqdm(range(num_samples)):
            generate_mmc_solutions(i)



def generate_trajectory(filepath):

    env = sogym(mode='train',observation_type='box_dense',vol_constraint_type='soft')
    obs=env.reset()

    def find_endpoints(x_center,y_center,L,theta):
        x_1=x_center-L*np.cos(theta)
        x_2=x_center+L*np.cos(theta)
        y_1=y_center-L*np.sin(theta)
        y_2=y_center+L*np.sin(theta)

        x_max = env.xmax[0]
        x_min = env.xmin[0]

        y_max = env.xmax[1]
        y_min = env.xmin[1]
        #clip x and y between min and max:
        x_1 = np.clip(x_1,x_min,x_max)
        x_2 = np.clip(x_2,x_min,x_max)
        y_1 = np.clip(y_1,y_min,y_max)
        y_2 = np.clip(y_2,y_min,y_max)
        return np.array([x_1[0],y_1[0],x_2[0],y_2[0]])

    def variables_2_actions(endpoints):
        return (2*endpoints - (env.xmax.squeeze()+env.xmin.squeeze()))/(env.xmax.squeeze()-env.xmin.squeeze())


    # The data point in json is in the dataset/dataset.zip file:
    # Let's open it and load it:
    
    data = load_top(filepath)
    filename = filepath.split('/')[-1].split('.')[0]
    design_variables= data['design_variables']
    # We define a new beta vector:
    load_vector = np.zeros((4,5))
    # I will define a volfrac vector which will be a 1 x 1 vector containing 'volfrac'.
    volfrac_vector = np.zeros((1,1))
    # I want to fill the vectors depending on the number of loads and supports I have:
    # Define the support vector with information about different supports
    support_vector = np.array([
        data['boundary_conditions']['selected_boundary'],
        data['boundary_conditions']['support_type'],
        data['boundary_conditions']['boundary_length'],
        data['boundary_conditions']['boundary_position']
    ])
    for i in range(data['boundary_conditions']['n_loads']):
        load_vector[0,i]=data['boundary_conditions']['load_position'][i]
        load_vector[1,i]=data['boundary_conditions']['load_orientation'][i]
        load_vector[2,i]=data['boundary_conditions']['magnitude_x'][i]
        load_vector[3,i]=data['boundary_conditions']['magnitude_y'][i]

    volfrac_vector = data['boundary_conditions']['volfrac']
    domain_vector = np.array([data['dx'],data['dy']])
    # Let's concatenate everything into a single vector 'beta':
    beta = np.concatenate((support_vector.flatten(order='F'),load_vector.flatten(order='F'),volfrac_vector,domain_vector),axis=None) # The new beta vector is a 25 x 1 vector
    
    components=np.split(np.array(design_variables),8)
    all_permutations = list(permutations(components))
    selected_permutations = random.sample(all_permutations, 10)
    trajectories = []
    for perm_idx, component_order in enumerate(selected_permutations):
        action_count = 0
        volume = 0
        n_steps_left = 1
        out_design_variables = np.zeros((8*6,1))
        variables_plot = []

        for single_component in component_order:
            endpoints = find_endpoints(single_component[0], single_component[1], single_component[2], single_component[5])
            # we add the two thicknesses to the numpy array:
            endpoints = np.append(endpoints, single_component[3])
            endpoints = np.append(endpoints, single_component[4])
            
            out_dict = {
                "design_variables": out_design_variables.tolist(),
                "volume": volume,
                "n_steps_left": n_steps_left,
                "conditions": beta.tolist(),
                "action": variables_2_actions(endpoints).tolist()
            }
            trajectories.append(out_dict)

            variables_plot.append(single_component.squeeze())

            # Replace the ith+6 values of out_design_variables with the single_component:
            out_design_variables[action_count*6:(action_count+1)*6] = single_component.reshape((6,1))
            action_count += 1
            n_steps_left = (8-action_count)/8
            #Get the volume after
            H, Phimax, allPhi, den = build_design(np.array(variables_plot).T)
            volume = sum(den)*env.EW*env.EH/(env.dx*env.dy)
        
        final_out_dict = {
            "design_variables": out_design_variables.tolist(),
            "volume": volume,
            "n_steps_left": 0,
            "conditions": beta.tolist(),
            "action": []
        }
        trajectories.append(final_out_dict)

    # Save the trajectories list using pickle
    with open(f'dataset/trajectories/{filename}_trajectories.pkl', 'wb') as fp:
        pickle.dump(trajectories, fp)



def generate_all(num_processes = 4):

    files = glob.glob('dataset/topologies/mmc/*.json')
    print(len(files))

    # Define the number of processes

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Wrap the files list with tqdm for a progress bar
        # Use executor.map with the wrapped files list
        list(tqdm(executor.map(generate_trajectory, files), total=len(files), desc="Generating Trajectories"))


import glob
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def list_unique_timestamps(folder='dataset/trajectories'):
    files = glob.glob('{}/*_trajectories.pkl'.format(folder))
    files_key = [file.split('/')[-1].split('_trajectories')[0] for file in files]
    return files_key

def load_trajectory(filename):
    trajectories = []
    with open('dataset/trajectories/{}_trajectories.pkl'.format(filename), 'rb') as f:
        all_permutation_data = pickle.load(f)

    trajectory_size = 9
    for i in range(0, len(all_permutation_data), trajectory_size):
        permutation_trajectory = all_permutation_data[i:i+trajectory_size]
        observations = []
        actions = []
        infos = []

        for data in permutation_trajectory:
            obs = np.concatenate((
                np.array(data['conditions']),
                np.array([data['volume']]),
                np.array([data['n_steps_left']]),
                np.array(data['design_variables']).squeeze()
            ))

            acts = np.array(data['action'])

            observations.append(obs)
            infos.append({})

            if len(acts) != 0:
                actions.append(acts.squeeze())

        terminal = True
        trajectories.append(Trajectory(obs=np.array(observations), acts=np.array(actions), terminal=terminal, infos=None))

    return trajectories

def load_all_trajectories(n_workers=4):
    filenames = list_unique_timestamps()
    trajectories = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(load_trajectory, filenames), total=len(filenames), desc="Loading trajectories"))
    
    for traj in results:
        trajectories.extend(traj)
    
    return trajectories
