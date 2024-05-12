# Standard library imports
import codecs
import datetime
import glob
import json
import multiprocessing as mp
import os
import pickle
import random
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import permutations
from multiprocessing import Pool, cpu_count
from functools import partial

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium import spaces

# Local module imports from sogym
from sogym.env import sogym
from sogym.mmc_optim import run_mmc
from sogym.struct import build_design

# Local module imports from imitation
from imitation.data.types import Trajectory


# Let's load an expert sample:
def count_top (filepath):
    """
    Count the number of .json files in the specified directory.

    Args:
        filepath (str): Path to the directory.

    Returns:
        int: Number of .json files in the directory.
    """
    path, dirs, files = next(os.walk(filepath))
    file_count = len(files)
    return file_count

def load_top (filepath):
    """
    Load a .json file from the specified filepath.

    Args:
        filepath (str): Path to the .json file.

    Returns:
        dict: Loaded JSON data as a dictionary.
    """
    obj_text = codecs.open(filepath, 'r', encoding='utf-8').read()
    dict = json.loads(obj_text)

    return dict

def generate_mmc_solutions(key,dataset_folder):
    """
    Generate a single solution for the MMC problem and save it in the dataset folder.

    Args:
        key (int): The key for multiprocessing.
        dataset_folder (str): Path to the dataset folder.

    Returns:
        None: The function saves the solution in the dataset folder.
    """
    env = sogym(mode='train',observation_type='dense',vol_constraint_type='soft',seed=key,resolution=50)
    obs = env.reset()
    xval, f0val, num_iter, H, Phimax, allPhi, den,N, cfg =  run_mmc(env.conditions,env.nelx,env.nely,env.dx,env.dy,plotting='nothing',verbose=0)
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
    """
    Generate a dataset of solutions for the MMC problem and save it in the dataset folder.

    Args:
        dataset_folder (str): Path to the dataset folder.
        num_threads (int, optional): Number of threads to use for multiprocessing. Defaults to 1.
        num_samples (int, optional): Number of samples to generate. Defaults to 10.
    """
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
    """
    Generate a trajectory from the specified JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        None: The function saves the generated trajectory as a pickle file.
    """
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
    """
    Generate trajectories for all JSON files in the specified directory using multiprocessing.

    Args:
        num_processes (int, optional): Number of processes to use for multiprocessing. Defaults to 4.
    """
    files = glob.glob('dataset/topologies/mmc/*.json')
    print("Length:",len(files))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        list(tqdm(executor.map(generate_trajectory, files), total=len(files), desc="Generating Trajectories"))


def list_unique_timestamps(folder='dataset/trajectories'):
    """
    List unique timestamps from the trajectory files in the specified folder.

    Args:
        folder (str, optional): Path to the folder containing trajectory files. Defaults to 'dataset/trajectories'.

    Returns:
        list: List of unique timestamps.
    """
    files = glob.glob('{}/*_trajectories.pkl'.format(folder))
    files_key = [file.split('/')[-1].split('_trajectories')[0] for file in files]
    return files_key

def load_trajectory(filename):
    """
    Load a trajectory from the specified file.

    Args:
        filename (str): Name of the trajectory file (without extension).

    Returns:
        list: List of Trajectory objects.
    """
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
    """
    Load all trajectories using multithreading.

    Args:
        n_workers (int, optional): Number of worker threads to use. Defaults to 4.

    Returns:
        list: List of all loaded trajectories.
    """
    filenames = list_unique_timestamps()
    trajectories = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(load_trajectory, filenames), total=len(filenames), desc="Loading trajectories"))
    
    for traj in results:
        trajectories.extend(traj)
    
    return trajectories

def find_endpoints(env, x_center, y_center, L, theta):
    x_1 = x_center - L * np.cos(theta)
    x_2 = x_center + L * np.cos(theta)
    y_1 = y_center - L * np.sin(theta)
    y_2 = y_center + L * np.sin(theta)

    x_max = env.xmax[0]
    x_min = env.xmin[0]

    y_max = env.xmax[1]
    y_min = env.xmin[1]
    # Clip x and y between min and max
    x_1 = np.clip(x_1, x_min, x_max)
    x_2 = np.clip(x_2, x_min, x_max)
    y_1 = np.clip(y_1, y_min, y_max)
    y_2 = np.clip(y_2, y_min, y_max)
    return np.array([x_1[0], y_1[0], x_2[0], y_2[0]])

def variables_2_actions(env, endpoints):
    return (2 * endpoints - (env.xmax.squeeze() + env.xmin.squeeze())) / (env.xmax.squeeze() - env.xmin.squeeze())

def process_file(env_kwargs, plot_terminated, filename, directory_path, num_permutations):
    try:
        env = sogym(**env_kwargs) if env_kwargs else sogym()
        obs,info = env.reset()

        file_path = os.path.join(directory_path, filename)
        mmc_solution = load_top(file_path)
        start_dict = {
            'dx': mmc_solution['dx'],
            'dy': mmc_solution['dy'],
            'nelx': mmc_solution['nelx'],
            'nely': mmc_solution['nely'],
            'conditions': mmc_solution['boundary_conditions']
        }

        design_variables = mmc_solution['design_variables']
        components = np.split(np.array(design_variables), mmc_solution['number_components'])

        results = []

        if num_permutations is None:
            num_permutations = 1

        for _ in range(num_permutations):
            obs,info = env.reset(start_dict=start_dict)

            if isinstance(env.observation_space, spaces.Dict):
                expert_observations = {key: [value] for key, value in obs.items()}
            else:
                expert_observations = [obs]

            expert_actions = []

            for single_component in components:
                endpoints = find_endpoints(env, single_component[0], single_component[1], single_component[2], single_component[5])
                endpoints = np.append(endpoints, single_component[3])
                endpoints = np.append(endpoints, single_component[4])

                action = variables_2_actions(env, endpoints)
                expert_actions.append(action)

                obs, reward, terminated, truncated, info = env.step(action)

                if isinstance(env.observation_space, spaces.Dict):
                    for key, value in obs.items():
                        expert_observations[key].append(value)
                else:
                    expert_observations.append(obs)

                if terminated and plot_terminated:
                    fig = env.plot(train_viz=False, axis=True)

            if isinstance(env.observation_space, spaces.Dict):
                expert_observations = {key: np.array(value[:-1]) for key, value in expert_observations.items()}
            else:
                expert_observations = np.array(expert_observations[:-1])

            expert_actions = np.array(expert_actions)

            results.append((expert_observations, expert_actions))
            np.random.shuffle(components)

        return results
    except Exception as e:
        # Log the error and return an empty list or None
        print(f"Error processing {filename}: {str(e)}")
        return None  # Indicates failure





def generate_expert_dataset(directory_path, env_kwargs=None,observation_type = 'topopt_game', plot_terminated=False, num_processes=None, num_permutations=1, file_fraction=1.0, chunk_size=1000):
    if num_processes is None:
        num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes, maxtasksperchild=10)

    file_list = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith(".json")]

    random.shuffle(file_list)

    num_files_to_process = int(len(file_list) * file_fraction)
    selected_files = file_list[:num_files_to_process]

    process_file_partial = partial(process_file, env_kwargs, plot_terminated, directory_path=directory_path, num_permutations=num_permutations)

    chunk_counter = 0

    try:
        with tqdm(total=len(selected_files), desc="Processing files", unit="file") as pbar:
            results = pool.imap_unordered(process_file_partial, selected_files)
            expert_observations_list = []
            expert_actions_list = []

            for result in results:
                if result is not None:
                    for expert_observations, expert_actions in result:
                        expert_observations_list.append(expert_observations)
                        expert_actions_list.append(expert_actions)

                        if len(expert_observations_list) >= chunk_size:
                            if isinstance(sogym(**env_kwargs).observation_space, spaces.Dict):
                                expert_observations = {key: np.concatenate([obs[key] for obs in expert_observations_list]) for key in expert_observations_list[0].keys()}
                            else:
                                expert_observations = np.concatenate(expert_observations_list)

                            expert_actions = np.concatenate(expert_actions_list)

                            # Determine the permutation part of the filename
                            if num_permutations is None:
                                perm_str = 'noperm'
                            else:
                                perm_str = f"{num_permutations}perm"
                            filename = f"dataset/expert/expert_dataset_{observation_type}_{perm_str}_chunk_{chunk_counter}.pkl"

                            # Save the data using pickle
                            with open(filename, 'wb') as f:
                                pickle.dump({'expert_observations': expert_observations, 'expert_actions': expert_actions}, f, protocol=4)
                                f.flush()
                                os.fsync(f.fileno())

                            expert_observations_list = []
                            expert_actions_list = []
                            chunk_counter += 1

                pbar.update(1)

            # Save any remaining data
            if len(expert_observations_list) > 0:
                if isinstance(sogym(**env_kwargs).observation_space, spaces.Dict):
                    expert_observations = {key: np.concatenate([obs[key] for obs in expert_observations_list]) for key in expert_observations_list[0].keys()}
                else:
                    expert_observations = np.concatenate(expert_observations_list)

                expert_actions = np.concatenate(expert_actions_list)

                filename = f"dataset/expert/expert_dataset_{observation_type}_{perm_str}_chunk_{chunk_counter}.pkl"

                with open(filename, 'wb') as f:
                    pickle.dump({'expert_observations': expert_observations, 'expert_actions': expert_actions}, f, protocol=4)
                    f.flush()
                    os.fsync(f.fileno())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pool.terminate()
