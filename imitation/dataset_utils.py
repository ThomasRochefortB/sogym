from sogym.env import sogym
import numpy as np
import json
from sogym.struct import build_design

def generate_trajectory(key, nelx=100, nely=50):

    env = sogym()
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
    
    f = open('dataset/dataset/data_'+str(key)+'.json',)
    data = json.load(f)
    
    design_variables= data['xval']
    
    fixednodes=data['fixednode']
    loadnode=data['loadnode']
    load_mat_print=np.zeros(((nely+1)*(nelx+1),1))
    load_mat_print[data['fixednode']]=1
    load_mat_print[data['loadnode'],0]=2
    load_mat_print=load_mat_print.reshape((nely+1,nelx+1,1),order='F')
    load_coords=np.argwhere(load_mat_print==2)[0][0:2]
    fixed_coord1=np.argwhere(load_mat_print==1)[0][0:2]
    fixed_coord2=np.argwhere(load_mat_print==1)[-1][0:2]
    volfrac=np.array(data['volfrac'])
    magnitude_x=data['magnitude_x'][0]
    magnitude_y=data['magnitude_y'][0]
    
    
    #Generating the beta vector:
    beta=np.array([load_coords[0]/(nely+1),  #Relative y position of load
                              load_coords[1]/(nelx+1),  #Relative x position of load
                              fixed_coord1[0]/(nely+1), #Relative y position of support_1
                            fixed_coord1[1]/(nelx+1),   #Relative x position of support_1
                              fixed_coord2[0]/(nely+1), #Relative y position of support_2
                              fixed_coord2[1]/(nelx+1), #Relative x position of support_2
                              volfrac,                  #Volume fraction (between 0 and 1)
                              magnitude_x,              #magnitude of load in x 
                              magnitude_y,              #magnitude of load in y 
                             ])   
    
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
        with open('dataset/trajectories/data_{}_{}.json'.format(key,action_count), 'w') as fp:
            json.dump(out_dict, fp)

        variables_plot.append(single_component.squeeze())

        # Replace the ith+6 values of out_design_variables with the single_component:
        out_design_variables[action_count*6:(action_count+1)*6]=single_component
        action_count+=1
        n_steps_left=(8-action_count)/8
        #Get the volume after
        H, Phimax,allPhi, den=build_design(np.array(variables_plot).T)
        volume = sum(den)*env.EW*env.EH/(env.DW*env.DH)
    
        final_out_dict={
            "design_variables":out_design_variables.tolist(),
            "volume":volume,
            "n_steps_left":0,
            "conditions":beta.tolist(),
            "action":[]
        }
    with open('dataset/trajectories/data_{}_{}.json'.format(key,8), 'w') as fp:
            json.dump(final_out_dict, fp)



def generate_all():
     # Get a list of all files in the dataset/dataset folder:
    import os
    import glob
    files = glob.glob('dataset/dataset/*.json')
    files_key=[int(file.split('_')[1].split('.')[0]) for file in files]

    print(len(files))
    for i in (files_key):
        generate_trajectory(i)

