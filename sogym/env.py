import gymnasium as gym
from gymnasium import spaces
from sogym.struct import build_design, calculate_compliance, calculate_strains, calculate_stresses
from sogym.rand_bc import generate_prompt
import numpy as np
import random
from sogym.rand_bc import gen_randombc
import cv2
import torch
import math
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Class defining the Structural Optimization Gym environment (so-gym):
class sogym(gym.Env):

    def __init__(self,N_components=8,resolution = 100, observation_type = 'dense',
                 mode = 'train',img_format='CHW',check_connectivity = False, 
                 seed=None,vol_constraint_type='hard',model=None,tokenizer=None):
     
        self.N_components = N_components
        self.mode = mode
        self.observation_type = observation_type
        self.img_format = img_format
        self.vol_constraint_type = vol_constraint_type
        self.check_connectivity = check_connectivity
        self.seed = seed
        self.N_actions = 6 
        self.counter=0  
        self.resolution = resolution
        self.fig = plt.figure(dpi=100)
        self.image_resolution = 64
        # series of render color for the plot function
        self.render_colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']

        self.action_space = spaces.Box(low=-1,high=1,shape=(self.N_actions,), dtype=np.float32)
        if self.img_format == 'CHW':
            img_shape = (3,self.image_resolution,self.image_resolution)
        elif self.img_format == 'HWC':
            img_shape = (self.image_resolution,self.image_resolution,3)

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
                                            "von_mises_stress":spaces.Box(0,255,img_shape,dtype=np.uint8) 
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
        seed = self.seed
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
        self.xmin=np.vstack((0, 0, 0.0, 0.0, 0.01, 0.01))  # (xa_min,ya_min, xb_min, yb_min, t1_min, t2_min)
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

        if self.observation_type == 'image':
            if self.img_format == 'CHW':
                empty_von_mises_stress = np.zeros((3, self.image_resolution, self.image_resolution), dtype=np.uint8)
            elif self.img_format == 'HWC':
                empty_von_mises_stress = np.zeros((self.image_resolution, self.image_resolution, 3), dtype=np.uint8)

            self.observation = {"image": self.gen_image(resolution=(self.image_resolution, self.image_resolution)),
                                "beta": np.float32(self.beta),
                                "design_variables": np.float32(self.variables.flatten()),
                                "volume": np.array([0.0], dtype=np.float32),
                                "n_steps_left": np.array([(self.N_components - self.action_count) / self.N_components], dtype=np.float32),
                                "von_mises_stress": empty_von_mises_stress}
            
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
        # The reward function is the sparse reward given only at the end of the episode if the desired volume fraction is respected.
        if self.action_count<self.N_components: # We are not at the end of the episode
            reward=0.0
            terminated=False

            if self.observation_type=='image':
                
                self.compliance, self.volume, self.U, self.F = calculate_compliance(
                self.H, self.conditions, self.dx, self.dy, self.nelx, self.nely)  # Calculate compliance, volume, and stress
                # Calculate strain and stress fields
                nDof = self.U.shape[0]  # Total number of DOFs
                nNod = nDof // 2       # Number of nodes

                # Create boolean masks for x-displacement and y-displacement DOFs
                mask_x = np.arange(nDof) % 2 == 0
                mask_y = np.arange(nDof) % 2 == 1

                # Extract x-displacement and y-displacement DOFs
                x_dofs = self.U[mask_x]
                y_dofs = self.U[mask_y]

                # Reshape displacement fields to match the structural domain grid
                x_displacement = x_dofs.reshape((self.nely+1, self.nelx+1), order='F')
                y_displacement = y_dofs.reshape((self.nely+1, self.nelx+1), order='F')

                strain_xx, strain_yy, strain_xy = calculate_strains(x_displacement, y_displacement)

                stress_xx, stress_yy, stress_xy = calculate_stresses(strain_xx, strain_yy, strain_xy)
                # Calculate von Mises stress
                self.von_mises_stress = np.sqrt(stress_xx**2 - stress_xx*stress_yy + stress_yy**2 + 3*stress_xy**2)

        else:  # We are at the end of the episode
            terminated = True
            self.last_Phi = self.Phi
            self.last_conditions, self.last_nelx, self.last_nely, self.last_x, self.last_y, self.last_dx, self.last_dy = \
                self.conditions, self.nelx, self.nely, self.x, self.y, self.dx, self.dy
            self.compliance, self.volume, self.U, self.F = calculate_compliance(
                self.H, self.conditions, self.dx, self.dy, self.nelx, self.nely)  # Calculate compliance, volume, and stress
             # Calculate strain and stress fields
            nDof = self.U.shape[0]  # Total number of DOFs
            nNod = nDof // 2       # Number of nodes

            # Create boolean masks for x-displacement and y-displacement DOFs
            mask_x = np.arange(nDof) % 2 == 0
            mask_y = np.arange(nDof) % 2 == 1

            # Extract x-displacement and y-displacement DOFs
            x_dofs = self.U[mask_x]
            y_dofs = self.U[mask_y]

            # Reshape displacement fields to match the structural domain grid
            x_displacement = x_dofs.reshape((self.nely+1, self.nelx+1), order='F')
            y_displacement = y_dofs.reshape((self.nely+1, self.nelx+1), order='F')

            strain_xx, strain_yy, strain_xy = calculate_strains(x_displacement, y_displacement)

            stress_xx, stress_yy, stress_xy = calculate_stresses(strain_xx, strain_yy, strain_xy)
            # Calculate von Mises stress
            self.von_mises_stress = np.sqrt(stress_xx**2 - stress_xx*stress_yy + stress_yy**2 + 3*stress_xy**2)

            # Check if connectivity check is required
            if self.check_connectivity:
                is_connected = self.check_connec()
            else:
                is_connected = True  # Assume connectivity check is not required or passed

            if self.vol_constraint_type == 'hard':
                if self.volume <= self.conditions['volfrac'] and is_connected:
                    denominator = ((self.compliance / len(self.conditions['loaddof_x'])))
                    denominator = np.log(denominator)
                    reward = (1 / denominator)
                else:
                    reward = 0.0
            else:
                if is_connected:
                    denominator = ((self.compliance / len(self.conditions['loaddof_x'])))
                    denominator = np.log(denominator)
                    reward = (1 / denominator ) * \
                            (1 - abs(self.volume - self.conditions['volfrac']))**2
                else:
                    reward = 0.0

            if math.isnan(reward):  # Catch potential NaN rewards
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
                 
      
        elif self.observation_type == 'image':
            # Use masked arrays to hide regions without material
            von_mises_stress_masked = np.ma.masked_where(self.H.reshape((self.nely+1, self.nelx+1), order='F') == 0, self.von_mises_stress)

            # Normalize the von Mises stress field to the range [0, 255]
            von_mises_stress_normalized = (von_mises_stress_masked - von_mises_stress_masked.min()) / (von_mises_stress_masked.max() - von_mises_stress_masked.min() + 1e-8)
            von_mises_stress_normalized = (von_mises_stress_normalized * 255).astype(np.uint8)

            # Fill the masked regions with a specific value (e.g., 255 for white)
            von_mises_stress_normalized = np.ma.filled(von_mises_stress_normalized, 255)

            # Resize the von Mises stress field to the desired resolution
            von_mises_stress_resized = cv2.resize(von_mises_stress_normalized, dsize=(self.image_resolution, self.image_resolution), interpolation=cv2.INTER_CUBIC)

            # Reshape the von Mises stress field to match the image format (e.g., CHW)
            if self.img_format == 'CHW':
                von_mises_stress_image = np.expand_dims(von_mises_stress_resized, axis=0)
                von_mises_stress_image = np.repeat(von_mises_stress_image, 3, axis=0)  # Repeat to create RGB channels
            elif self.img_format == 'HWC':
                von_mises_stress_image = np.expand_dims(von_mises_stress_resized, axis=-1)
                von_mises_stress_image = np.repeat(von_mises_stress_image, 3, axis=-1)  # Repeat to create RGB channels

            self.observation = {"image": self.gen_image(resolution=(self.image_resolution, self.image_resolution)),
                                "beta": np.float32(self.beta),
                                "design_variables": np.float32(self.variables.flatten()) / np.pi,
                                "volume": np.array([self.volume], dtype=np.float32),
                                "n_steps_left": np.array([(self.N_components - self.action_count) / self.N_components], dtype=np.float32),
                                "von_mises_stress": von_mises_stress_image}

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
 
    def plot(self, train_viz=True, axis=True):
        plt.rcParams["figure.figsize"] = (10,10)        

        if train_viz:
            dx, dy, nelx, nely, x, y, condition_dict, Phi = (
                self.last_dx, self.last_dy, self.last_nelx, self.last_nely,
                self.last_x, self.last_y, self.last_conditions, self.last_Phi
            )
        else:
            dx, dy, nelx, nely, x, y, condition_dict, Phi = (
                self.dx, self.dy, self.nelx, self.nely,
                self.x, self.y, self.conditions, self.Phi
            )

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if self.variables_plot == [] and not train_viz:
            color = ['white']
        else:
            color = self.render_colors

        for i, c in zip(range(Phi.shape[1]), color):
            ax.contourf(x, y, Phi[:, i].reshape((nely + 1, nelx + 1), order='F'), [0, 1], colors=[c])

        ax.add_patch(plt.Rectangle((0, 0), dx, dy, clip_on=False, linewidth=1, fill=False))

        boundary_conditions = [
            (0.0, 'left', (0.0, dy * condition_dict['boundary_position']), 90),
            (0.25, 'right', (dx + 0.1, dy * condition_dict['boundary_position']), 90),
            (0.5, 'bottom', (dx * condition_dict['boundary_position'], dy), 0),
            (0.75, 'top', (dx * condition_dict['boundary_position'], -0.1), 0)
        ]

        for boundary, name, xy, angle in boundary_conditions:
            if condition_dict['selected_boundary'] == boundary:
                ax.add_patch(plt.Rectangle(
                    xy=xy,
                    width=condition_dict['boundary_length'] * (dy if name in ['left', 'right'] else dx),
                    height=0.1,
                    angle=angle,
                    hatch='/',
                    clip_on=False,
                    linewidth=0
                ))

                for i in range(condition_dict['n_loads']):
                    load_pos = dy * condition_dict['load_position'][i] if name in ['left', 'right'] else dx * condition_dict['load_position'][i]
                    arrow_pos = (
                        (dx - condition_dict['magnitude_x'][i] * 0.2, load_pos - condition_dict['magnitude_y'][i] * 0.2)
                        if name == 'left' else
                        (0.0 - condition_dict['magnitude_x'][i] * 0.2, load_pos - condition_dict['magnitude_y'][i] * 0.2)
                        if name == 'right' else
                        (load_pos - condition_dict['magnitude_x'][i] * 0.2, -condition_dict['magnitude_y'][i] * 0.2)
                        if name == 'bottom' else
                        (load_pos - condition_dict['magnitude_x'][i] * 0.2, dy - condition_dict['magnitude_y'][i] * 0.2)
                    )
                    ax.arrow(*arrow_pos,
                            dx=condition_dict['magnitude_x'][i] * 0.2,
                            dy=condition_dict['magnitude_y'][i] * 0.2,
                            width=0.2 / 8,
                            length_includes_head=True,
                            head_starts_at_zero=False)

        if not axis:
            ax.set_axis_off()

        return self.fig


    def gen_image(self, resolution):
        self.plot(train_viz=False, axis=False)  # Pass the figure object to plot
        self.fig.tight_layout(pad=0)

        # Convert the figure to a numpy array without rendering it to the screen
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)

        # Close the figure to free up memory
        plt.close(self.fig)

        # Resize the image to the desired resolution using OpenCV
        res = cv2.resize(buf, dsize=(resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        # Convert res to channel first if needed
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

        # Define slices for load based on the opposite boundary and load_position
        load_slices = []
        for i in range(self.conditions['n_loads']):
            load_pos_y = min(int(self.conditions['load_position'][i] * self.nely), self.nely - 1)
            load_pos_x = min(int(self.conditions['load_position'][i] * self.nelx), self.nelx - 1)

            if opposite_boundary_key == 0:  # Left
                load_slices.append((load_pos_y, 0))
            elif opposite_boundary_key == 1:  # Right
                load_slices.append((load_pos_y, self.nelx - 1))
            elif opposite_boundary_key == 2:  # Bottom
                load_slices.append((self.nely - 1, load_pos_x))
            elif opposite_boundary_key == 3:  # Top
                load_slices.append((0, load_pos_x))
        # Get labels for support and loads
        labels_support = labels[boundary_slices[boundary_key]]
        for idx in load_slices:
            labels_load.append(labels[idx])

        # Check connection between load labels and support label
        connec = [load in labels_support and load != 0 for load in labels_load]


        # Return True if all connections are True (and not empty)
        return bool(connec) and all(connec)

    
