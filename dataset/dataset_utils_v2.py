from sogym.env import sogym
import numpy as np
import json
from sogym.struct import build_design

def generate_trajectory(key):

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
    
    f = open('dataset/raw_data_v2/data_'+str(key)+'.json',)
    data = json.load(f)
    design_variables= data['xval']
    # We define a new beta vector:
    load_vector = np.zeros((4,5))
    # I will define a volfrac vector which will be a 1 x 1 vector containing 'volfrac'.
    volfrac_vector = np.zeros((1,1))
    # I want to fill the vectors depending on the number of loads and supports I have:
    # Define the support vector with information about different supports
    support_vector = np.array([
        data['conditions']['selected_boundary'],
        data['conditions']['support_type'],
        data['conditions']['boundary_length'],
        data['conditions']['boundary_position']
    ])
    for i in range(data['conditions']['n_loads']):
        load_vector[0,i]=data['conditions']['load_position'][i]
        load_vector[1,i]=data['conditions']['load_orientation'][i]
        load_vector[2,i]=data['conditions']['magnitude_x'][i]
        load_vector[3,i]=data['conditions']['magnitude_y'][i]

    volfrac_vector = data['conditions']['volfrac']
    # Let's concatenate everything into a single vector 'beta':
    beta = np.concatenate((support_vector.flatten(order='F'),load_vector.flatten(order='F'),volfrac_vector),axis=None) # The new beta vector is a 25 x 1 vector
    
    components=np.split(np.array(design_variables),8)
    action_count=0
    volume=0
    n_steps_left=1
    out_design_variables=np.zeros((8*6,1))
    variables_plot=[]

    for single_component in components:
        endpoints = find_endpoints(single_component[0],single_component[1],single_component[2],single_component[5])
        # we add the two thicknesses to the numpy array:
        endpoints=np.append(endpoints,single_component[3])
        endpoints=np.append(endpoints,single_component[4])        
        out_dict={
            "design_variables":out_design_variables.tolist(),
            "volume":volume,
            "n_steps_left":n_steps_left,
            "conditions":beta.tolist(),
            "action":variables_2_actions(endpoints).tolist()
        }
        with open('dataset/trajectories_v2/data_{}_{}.json'.format(key,action_count), 'w') as fp:
            json.dump(out_dict, fp)

        variables_plot.append(single_component.squeeze())

        # Replace the ith+6 values of out_design_variables with the single_component:
        out_design_variables[action_count*6:(action_count+1)*6]=single_component.reshape((6,1))
        action_count+=1
        n_steps_left=(8-action_count)/8
        #Get the volume after
        H, Phimax,allPhi, den=build_design(np.array(variables_plot).T)
        volume = sum(den)*env.EW*env.EH/(env.dx*env.dy)
    
        final_out_dict={
            "design_variables":out_design_variables.tolist(),
            "volume":volume,
            "n_steps_left":0,
            "conditions":beta.tolist(),
            "action":[]
        }
    with open('dataset/trajectories_v2/data_{}_{}.json'.format(key,8), 'w') as fp:
            json.dump(final_out_dict, fp)



def generate_all():
     # Get a list of all files in the dataset/dataset folder:
    import os
    import glob
    files = glob.glob('dataset/raw_data_v2/*.json')
    files_key=[int(file.split('/')[-1].split('_')[1].split('.')[0]) for file in files]
    print(len(files))
    for i in (files_key):
        generate_trajectory(i)

