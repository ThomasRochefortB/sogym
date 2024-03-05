import gymnasium as gym
from gymnasium import spaces
from sogym.struct import build_design, calculate_compliance
from sogym.rand_bc import generate_prompt
import numpy as np
import random
import matplotlib.pyplot as plt
from sogym.rand_bc import gen_randombc
import cv2
import torch
import math
#Class defining the Structural Optimization Gym environment (so-gym):
class sogym(gym.Env):

    def __init__(self,N_components=8,resolution = 100, observation_type = 'dense',mode = 'train',img_format='CHW',vol_constraint_type='hard',model=None,tokenizer=None):
     
        self.N_components = N_components
        self.mode = mode
        self.observation_type = observation_type
        self.img_format = img_format
        self.vol_constraint_type = vol_constraint_type
        self.N_actions = 6 
        self.counter=0  
        self.resolution = resolution
        # series of render color for the plot function
        self.render_colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']

        self.action_space = spaces.Box(low=-1,high=1,shape=(self.N_actions,), dtype=np.float32)
        if self.img_format == 'CHW':
            img_shape = (3,128,128)
        elif self.img_format == 'HWC':
            img_shape = (128,128,3)

        if self.observation_type =='dense':
            self.observation_space = spaces.Dict(
                                        {
                                            "beta": spaces.Box(-1, 2.0, (27,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )
        
        elif self.observation_type =='box_dense':
            self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(27+1+1+self.N_components*self.N_actions,), dtype=np.float32) 


        elif self.observation_type =='image':
            self.observation_space = spaces.Dict(
                                        {
                                            "image": spaces.Box(0, 255, img_shape,dtype=np.uint8), # Image of the current design
                                            "beta": spaces.Box(-1, 2.0, (27,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )   
        elif self.observation_type =='text_dict':
            self.tokenizer = tokenizer
            self.model = model
            #device agnostic code:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.observation_space = spaces.Dict(
                                        spaces={
                                            # Prompt will have no max min (-inf,inf)
                                            "prompt": spaces.Box(-np.inf,np.inf, (768*512,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )
        else:
            raise ValueError('Invalid observation space type. Only "dense", "box_dense" , "text_dict"(experimental) and "image" are supported.')

    def reset(self,seed=None,start_dict=None):
        
        if self.mode == 'test':
            self.counter+=1
            self.dx, self.dy, self.nelx, self.nely, self.conditions = gen_randombc(seed=self.counter, resolution=self.resolution)
        else:
            self.dx, self.dy, self.nelx, self.nely, self.conditions = gen_randombc(seed=seed, resolution=self.resolution)
            
        if start_dict is not None:
            self.dx = start_dict['dx']
            self.dy = start_dict['dy']
            self.nelx = start_dict['nelx']
            self.nely = start_dict['nely']
            self.conditions = start_dict['conditions']

            
        if self.observation_type == 'text_dict':
            prompt = generate_prompt(self.conditions,self.dx,self.dy)
            self.model_output =  self.model(self.tokenizer(prompt, return_tensors="pt",padding = 'max_length').input_ids.to(self.device)).last_hidden_state.detach().cpu().numpy().flatten()
        self.EW=self.dx / self.nelx # length of element
        self.EH=self.dy/ self.nely # width of element     
        self.xmin=np.vstack((0, 0, 0.0, 0.0, 0.001, 0.001))  # (xa_min,ya_min, xb_min, yb_min, t1_min, t2_min)
        self.xmax=np.vstack((self.dx, self.dy, self.dx, self.dy, 0.05*min(self.dx,self.dy),0.05*min(self.dx,self.dy))) # (xa_max,ya_max, xb_max, yb_max, t1_max, t2_max)
        self.x,self.y=np.meshgrid(np.linspace(0, self.dx,self.nelx+1),np.linspace(0,self.dy,self.nely+1))                # coordinates of nodal points
        self.variables_plot=[]

        # We define a new beta vector:
        load_vector = np.zeros((4,5))
        # I will define a volfrac vector which will be a 1 x 1 vector containing 'volfrac'.
        volfrac_vector = np.zeros((1,1))
        # I want to fill the vectors depending on the number of loads and supports I have:
        # Define the support vector with information about different supports
        support_vector = np.array([
            self.conditions['selected_boundary'],
            self.conditions['support_type'],
            self.conditions['boundary_length'],

            self.conditions['boundary_position']
        ])
        for i in range(self.conditions['n_loads']):
            load_vector[0,i]=self.conditions['load_position'][i]
            load_vector[1,i]=self.conditions['load_orientation'][i]
            load_vector[2,i]=self.conditions['magnitude_x'][i]
            load_vector[3,i]=self.conditions['magnitude_y'][i]

        volfrac_vector = self.conditions['volfrac']

        # Let's define a 'domain vector' which will have dx and dy:
        domain_vector = np.array([self.dx,self.dy])

        # Let's concatenate everything into a single vector 'beta':
        self.beta = np.concatenate((support_vector.flatten(order='F'),load_vector.flatten(order='F'),volfrac_vector, domain_vector),axis=None) # The new beta vector is a 27 x 1 vector

    
        self.variables=np.zeros((self.N_components*self.N_actions,1))
        self.out_conditions=self.beta
        self.action_count=0
        self.saved_volume=[0.0]
        self.plot_conditions = self.out_conditions
        # I need to initialize an empty instance of Phi:
        self.Phi = np.zeros(((self.nelx+1)*(self.nely+1), self.N_components))
        if self.observation_type=='dense':
            self.observation={"beta":np.float32(self.beta),
                            "design_variables":np.float32(self.variables.flatten()),
                            "volume":np.array([0.0],dtype=np.float32),
                            "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                            }
        elif self.observation_type=='box_dense':        
            self.observation=np.concatenate(
                (np.float32(self.beta),
                 np.array([0.0],dtype=np.float32),
                 np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                 np.float32(self.variables.flatten()))
                 ,axis=0)

        elif self.observation_type=='image':
            self.observation={"image":self.gen_image(resolution=(128,128)),
                            "beta":np.float32(self.beta),
                            "design_variables":np.float32(self.variables.flatten()),
                            "volume":np.array([0.0],dtype=np.float32),
                            "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                            }
        elif self.observation_type == 'text_dict':
            self.observation = {
                "prompt": np.float32(self.model_output),
                "design_variables": np.float32(self.variables.flatten()),
                "volume": np.array([0.0], dtype=np.float32),
                "n_steps_left": np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
            }
        else:

            raise ValueError('Invalid observation space type. Only "dense" and "image" are supported.')
        info ={}
        return (self.observation,info )
        
        
    def step(self, action,evaluate=True):
        self.action_count+=1   # One action is taken
       
        self.new_variables=(self.xmax.squeeze()-self.xmin.squeeze())/(2)*(action-1)+self.xmax.squeeze()  # We convert from  [-1,1] to [xmin,xmax]

        # The design variables are infered from the two endpoints and the two thicknesses:
        x_center = (self.new_variables[0] + self.new_variables[2])/2
        y_center = (self.new_variables[1]+self.new_variables[3])/2
        L = np.sqrt((self.new_variables[0]-self.new_variables[2])**2 + (self.new_variables[1]-self.new_variables[3])**2  )/2
        t_1 = self.new_variables[4]
        t_2 =self.new_variables[5]
        theta =  np.arctan2(self.new_variables[3] - self.new_variables[1], self.new_variables[2]- self.new_variables[0])

        # We build a new design variable vector
        formatted_variables = np.array([x_center,y_center, L, t_1, t_2, theta])
        self.variables[(self.action_count-1)*self.N_actions:self.action_count*self.N_actions,0]= formatted_variables
        self.variables_plot.append(formatted_variables)

        # We build the topology with the new design variables:
        self.H, self.Phimax,self.Phi, den=build_design(np.array(self.variables_plot).T, self.dx,self.dy, self.nelx,self.nely)    # self.H is the Heaviside projection of the design variables and self.Phi are the design surfaces.        
        nEle = self.nelx*self.nely
        nNod=(self.nelx+1)*(self.nely+1)
        nodMat = np.reshape(np.array(range(0,nNod)),(1+self.nely,1+self.nelx),order='F')                    # maxtrix of nodes numbers (int32)
        edofVec = np.reshape(2*nodMat[0:-1,0:-1],(nEle,1),order='F')
        edofMat = edofVec + np.array([0, 1, 2*self.nely+2,2*self.nely+3, 2*self.nely+4, 2*self.nely+5, 2, 3])              # connectivity matrix
        eleNodesID = edofMat[:,0:8:2]/2   
        #FEA
        self.den=np.sum(self.H[eleNodesID.astype('int')],1)/4 
        self.volume=sum(self.den)*self.EW*self.EH/(self.dx*self.dy)
        
        truncated = False
        terminated = self.action_count >= self.N_components
        reward = 0.0

        if terminated:
            self.last_Phi = self.Phi
            self.last_conditions, self.last_nelx, self.last_nely, self.last_x, self.last_y, self.last_dx, self.last_dy = \
                self.conditions, self.nelx, self.nely, self.x, self.y, self.dx, self.dy

            if evaluate:
                self.compliance, self.volume, self.U, self.F = calculate_compliance(self.H, self.conditions, self.dx, self.dy, self.nelx, self.nely)
                
                if self.vol_constraint_type == 'hard' and self.volume <= self.conditions['volfrac'] and self.check_connec():
                    reward = 1 / (self.compliance / len(self.conditions['loaddof_x']) + 1e-8)
                elif self.vol_constraint_type != 'hard' and self.check_connec():
                    reward = (1 / (self.compliance / len(self.conditions['loaddof_x']) + 1e-8)) * (1 - abs(self.volume - self.conditions['volfrac']))**6
            else:
                self.compliance, self.volume, self.U, self.F = 0.0, 0.0, 0.0, 0.0

            if math.isnan(reward):   #Sometimes my reward is Nan ... Need to investigate later
                reward = 0.0

        info={}
        if self.observation_type=='dense':
            self.observation = {"beta":np.float32(self.beta),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                    "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                    }
        elif self.observation_type=='box_dense':        
            self.observation=np.concatenate(
                (np.float32(self.beta),
                 np.array([self.volume],dtype=np.float32),
                 np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                 np.float32(self.variables.flatten())/np.pi)
                 ,axis=0)
                 
        elif self.observation_type=='image':
            self.observation = {"image":self.gen_image(resolution=(128,128)),
                    "beta":np.float32(self.beta),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                    "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                    }
        elif self.observation_type == 'text_dict':
            self.observation = {
                "prompt": np.float32(self.model_output),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                "n_steps_left": np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
            }
        else:
            raise ValueError('Invalid observation space type. Only "dense" and "image" are supported.')
        self.saved_volume.append(self.volume)

        return self.observation, reward, terminated, truncated, info
    

    def plot(self, mode='human',test=None, train_viz=True):
        #plt.rcParams["figure.figsize"] = (5*self.dx,5*self.dy)        
        plt.rcParams["figure.figsize"] = (10,10)        
        if train_viz:
            dx = self.last_dx
            dy = self.last_dy
            nelx = self.last_nelx
            nely = self.last_nely
            x = self.last_x
            y = self.last_y
            condition_dict = self.last_conditions
        else:
            dx = self.dx
            dy = self.dy
            nelx = self.nelx
            nely = self.nely
            x = self.x
            y = self.y
            condition_dict = self.conditions

        fig = plt.figure()
        ax = plt.subplot(111)
        if train_viz:
            for i, color in zip(range(0,self.last_Phi.shape[1]), self.render_colors):
                ax.contourf(x,y,self.last_Phi[:,i].reshape((nely+1,nelx+1),order='F'),[0,1],colors=color)
        else:
            if self.variables_plot==[]:
                 for i, color in zip(range(0,self.Phi.shape[1]), self.render_colors):
                    ax.contourf(x,y,self.Phi[:,i].reshape((nely+1,nelx+1),order='F'),[0,1],colors='white')
            else:
                for i, color in zip(range(0,self.Phi.shape[1]), self.render_colors):
                    ax.contourf(x,y,self.Phi[:,i].reshape((nely+1,nelx+1),order='F'),[0,1],colors=color)
                    
            
        # Add a rectangle to show the domain boundary:
        ax.add_patch(plt.Rectangle((0,0),dx, dy,clip_on=False,linewidth = 1,fill=False))
        
        if condition_dict['selected_boundary']==0.0:  # Left boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (0.0,dy*(condition_dict['boundary_position'])),
                                    width = condition_dict['boundary_length']*dy, 
                                    height = 0.1,
                                    angle = 90,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(dx-(condition_dict['magnitude_x'][i]*0.2),dy*(condition_dict['load_position'][i])-condition_dict['magnitude_y'][i]*0.2,
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)
                
        elif condition_dict['selected_boundary']==0.25: # Right boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx+0.1,dy*(condition_dict['boundary_position'])),
                                    width = condition_dict['boundary_length']*dy, 
                                    height = 0.1,
                                    angle = 90,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(0.0-(condition_dict['magnitude_x'][i]*0.2),dy*(condition_dict['load_position'][i])-condition_dict['magnitude_y'][i]*0.2,
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)
        elif condition_dict['selected_boundary']==0.5: # Bottom boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx*self.conditions['boundary_position'],dy),
                                    width = self.conditions['boundary_length']*self.dx, 
                                    height = 0.1,
                                    angle = 0.0,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(dx*(condition_dict['load_position'][i])-condition_dict['magnitude_x'][i]*0.2,-(condition_dict['magnitude_y'][i]*0.2),
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)
                
        elif condition_dict['selected_boundary']==0.75: # Top boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx*condition_dict['boundary_position'],-0.1),
                                    width = condition_dict['boundary_length']*dx, 
                                    height = 0.1,
                                    angle = 0.0,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(dx*(condition_dict['load_position'][i])-condition_dict['magnitude_x'][i]*0.2,dy-(condition_dict['magnitude_y'][i]*0.2),
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)

        ax.set_axis_off()
        plt.close()
        return fig    

    def gen_image(self,resolution):
        fig = self.plot(train_viz=False)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        # Convert the canvas to an RGB numpy array
        w, h = fig.canvas.get_width_height()
        #print(w,h)
        # Save the figure as a png
        fig.savefig('test.png')
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #print(buf.shape)
        buf.shape = (h, w, 3)
        # Now we can save it to a numpy array.
        #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
        # Let's resize the image to something more reasonable using numpy:
        res = cv2.resize(buf, dsize=(resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        # Convert res to channel first:
        if self.img_format == 'CHW':
            res = np.moveaxis(res, -1, 0)
        return res

    def check_connec(self):
        # Load grayscale image and threshold to create a binary image
        img = self.den.reshape((self.nely, self.nelx), order='F')
        thresh = cv2.threshold(img, 0.1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Apply connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Initialize variables
        labels_load = []
        boundary_key = int(self.conditions['selected_boundary'] / 0.25)

        # Define opposite boundaries for each boundary_key
        opposite_boundaries = {
            0: 1,  # Left to Right
            1: 0,  # Right to Left
            2: 3,  # Bottom to Top
            3: 2   # Top to Bottom
        }
        opposite_boundary_key = opposite_boundaries[boundary_key]


        boundary_slices = [(slice(int(self.conditions['boundary_position'] * self.nely),
                                int((self.conditions['boundary_position'] + self.conditions['boundary_length']) * self.nely)),
                            0),
                        (slice(int(self.conditions['boundary_position'] * self.nely),
                                int((self.conditions['boundary_position'] + self.conditions['boundary_length']) * self.nely)),
                            -1),
                        (-1, slice(int(self.conditions['boundary_position'] * self.nelx),
                                    int((self.conditions['boundary_position'] + self.conditions['boundary_length']) * self.nelx))),
                        (0, slice(int(self.conditions['boundary_position'] * self.nelx),
                                    int((self.conditions['boundary_position'] + self.conditions['boundary_length']) * self.nelx)))]

        load_slices = []
        for i in range(self.conditions['n_loads']):
            if opposite_boundary_key == 0:  # Left
                load_slices.append((int(self.conditions['load_position'][i] * self.nely), 0))
            elif opposite_boundary_key == 1:  # Right
                load_slices.append((int(self.conditions['load_position'][i] * self.nely), -1))
            elif opposite_boundary_key == 2:  # Bottom
                load_slices.append((-1, int(self.conditions['load_position'][i] * self.nelx)))
            elif opposite_boundary_key == 3:  # Top
                load_slices.append((0, int(self.conditions['load_position'][i] * self.nelx)))

        # Get labels for support and loads
        labels_support = labels[boundary_slices[boundary_key]]
        for idx in load_slices:
            labels_load.append(labels[idx])

        # Check connection between load labels and support label
        connec = [load in labels_support and load != 0 for load in labels_load]


        # Return True if all connections are True (and not empty)
        return bool(connec) and all(connec)

    
