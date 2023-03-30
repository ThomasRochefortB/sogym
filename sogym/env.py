import gym
from sogym.struct import *
import numpy as np
import random
import matplotlib.pyplot as plt
from sogym.rand_bc import gen_randombc
import cv2

#Class defining the Structural Optimization Gym environment (so-gym):
class sogym(gym.Env):

    def __init__(self,N_components=8,observation_type = 'dense',mode = 'train',img_format='CHW',vol_constraint_type='hard'):
     
        self.N_components = N_components
        self.mode = mode
        self.observation_type = observation_type
        self.img_format = img_format
        self.vol_constraint_type = vol_constraint_type
       
        self.N_actions = 6        
        # series of render color for the plot function
        self.render_colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']

        self.action_space = gym.spaces.Box(low=-1,high=1,shape=(self.N_actions,), dtype=np.float32)
        if self.img_format == 'CHW':
            img_shape = (3,64,128)
        elif self.img_format == 'HWC':
            img_shape = (64,128,3)

        if self.observation_type =='dense':
            self.observation_space = gym.spaces.Dict(
                                        spaces={
                                            "beta": gym.spaces.Box(-1, 1, (25,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":gym.spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": gym.spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":gym.spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )
        
        elif self.observation_type =='box_dense':
                                         self.observation_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(25+1+1+self.N_components*self.N_actions,), dtype=np.float32) 


        elif self.observation_type =='image':
            #To define the image space here.
            self.observation_space = gym.spaces.Dict(
                                        spaces={
                                            "image": gym.spaces.Box(0, 255, img_shape,dtype=np.uint8), # Image of the current design
                                            "beta": gym.spaces.Box(-1, 1, (25,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":gym.spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": gym.spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":gym.spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )   
            pass
        else:
            raise ValueError('Invalid observation space type. Only "dense", "box_dense" and "image" are supported.')

    def reset(self):
        self.dx, self.dy, self.nelx, self.nely, self.conditions = gen_randombc()
        self.EW=self.dx / self.nelx # length of element
        self.EH=self.dy/ self.nely # width of element     
        self.xmin=np.vstack((0, 0, 0.0, 0.0, 0.0, 0.0))  # (xa_min,ya_min, xb_min, yb_min, t1_min, t2_min)
        self.xmax=np.vstack((self.dx, self.dy, self.dx, self.dy, 0.2, 0.2)) # (xa_max,ya_max, xb_max, yb_max, t1_max, t2_max)
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
        # Let's concatenate everything into a single vector 'beta':
        self.beta = np.concatenate((support_vector.flatten(order='F'),load_vector.flatten(order='F'),volfrac_vector),axis=None) # The new beta vector is a 25 x 1 vector

    
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
            self.observation={"image":self.gen_image(resolution=(128,64)),
                            "beta":np.float32(self.beta),
                            "design_variables":np.float32(self.variables.flatten()),
                            "volume":np.array([0.0],dtype=np.float32),
                            "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                            }
        else:
            raise ValueError('Invalid observation space type. Only "dense" and "image" are supported.')

        return self.observation 
        
        
    def step(self, action):
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
        

        # The reward function is the sparse reward given only at the end of the episode if the desired volume fraction is respected.
        if self.action_count<self.N_components: # We are not at the end of the episode
            reward=0.0
            done=False

        else: # We are at the end of the episode
            done=True
            self.last_Phi = self.Phi
            self.last_conditions,self.last_nelx, self.last_nely ,self.last_x, self.last_y ,self.last_dx, self.last_dy = self.conditions, self.nelx, self.nely, self.x, self.y, self.dx, self.dy
            self.compliance,self.volume, self.U, self.F=calculate_compliance(self.H,self.conditions,self.dx,self.dy,self.nelx,self.nely) # We calculate the compliance, volume and von Mises stress of the structure


            if self.vol_constraint_type=='hard':  
                if self.volume<= self.conditions['volfrac'] and self.check_connec(): # The desired volume fraction is respected
                    reward=(1/(self.compliance+1e-8)) # The reward is the inverse of the compliance (AKA the stiffness of the structure)
                else:
                    reward=0.0
            else:
                if self.check_connec():
                    reward=(1/(self.compliance+1e-8)) * (1-(self.volume-self.conditions['volfrac'])**2) # The reward is the inverse of the compliance (AKA the stiffness of the structure) times a penalty term for the volume fraction
                else:
                    reward=0.0
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
            self.observation = {"image":self.gen_image(resolution=(128,64)),
                    "conditions":np.float32(self.beta),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                    "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                    }
        else:
            raise ValueError('Invalid observation space type. Only "dense" and "image" are supported.')
        self.saved_volume.append(self.volume)

        return self.observation, reward, done, info
    

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
                ax.contourf(x,y,np.flipud(self.last_Phi[:,i].reshape((nely+1,nelx+1),order='F')),[0,1],colors=color)
        else:
            for i, color in zip(range(0,self.Phi.shape[1]), self.render_colors):
                ax.contourf(x,y,np.flipud(self.Phi[:,i].reshape((nely+1,nelx+1),order='F')),[0,1],colors=color)
        
                    # Add a rectangle to show the domain boundary:
        ax.add_patch(plt.Rectangle((0,0),dx, dy,
                                    clip_on=False,linewidth = 1,fill=False))
        
        if condition_dict['selected_boundary']==0.0:  # Left boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (0.0,dy*(1.0-condition_dict['boundary_position']-condition_dict['boundary_length'])),
                                    width = condition_dict['boundary_length']*dy, 
                                    height = 0.1,
                                    angle = 90,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(dx-(condition_dict['magnitude_x'][i]*0.2),dy*(1-condition_dict['load_position'][i]),
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)
                
        elif condition_dict['selected_boundary']==0.25: # Right boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx+0.1,dy*(1.0-condition_dict['boundary_position']-condition_dict['boundary_length'])),
                                    width = condition_dict['boundary_length']*dy, 
                                    height = 0.1,
                                    angle = 90,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(0.0-(condition_dict['magnitude_x'][i]*0.2),dy*(1-condition_dict['load_position'][i])-condition_dict['magnitude_y'][i]*0.2,
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)
        elif condition_dict['selected_boundary']==0.5: # Bottom boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx*self.conditions['boundary_position'],-0.1),
                                    width = self.conditions['boundary_length']*self.dx, 
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
                
        elif condition_dict['selected_boundary']==0.75: # Top boundary
            # Add a blue rectangle to show the support 
            ax.add_patch(plt.Rectangle(xy = (dx*condition_dict['boundary_position'],dy),
                                    width = condition_dict['boundary_length']*dx, 
                                    height = 0.1,
                                    angle = 0.0,
                                    hatch='/',
                                        clip_on=False,
                                        linewidth = 0))

            for i in range(condition_dict['n_loads']):
                ax.arrow(dx*(condition_dict['load_position'][i])-condition_dict['magnitude_x'][i]*0.2,0.0-(condition_dict['magnitude_y'][i]*0.2),
                            dx= condition_dict['magnitude_x'][i]*0.2,
                            dy = condition_dict['magnitude_y'][i]*0.2,
                            width=0.2/8,
                            length_includes_head=True,
                            head_starts_at_zero=False)

        ax.set_axis_off()

        plt.close()
        return fig    

    def gen_image(self,resolution=(128,64)):
        assert resolution[0] % resolution[1] == 0, "2x1 aspect ratio required"
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
        res = cv2.resize(buf, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)
        # Convert res to channel first:
        if self.img_format == 'CHW':
            res = np.moveaxis(res, -1, 0)
        return res

    def check_connec(self):
        connec=[]
        # Load grayscale image
        img = (self.den.reshape((self.nely,self.nelx),order='F'))
        # Threshold the image to create a binary image with dark pixels as 1s and light pixels as 0s
        thresh = cv2.threshold(img,0.1, 255, cv2.THRESH_BINARY)[1]
        thresh = np.array(thresh,dtype=np.uint8)

        # Apply a connected component analysis to find the connected components in the image
        output = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        # Let's say we want to know if the load and the support boundaries are connected:
        #Relative y position of load
        #Relative x position of load
        #Relative y position of support_1
        #Relative x position of support_1
        #Relative y position of support_2
        #Relative x position of support_2
        #Volume fraction (between 0 and 1)
        #magnitude of load in x 
        #magnitude of load in y 

        if self.conditions['selected_boundary']==0.0:  # Left boundary
            label_support = labels[int(self.conditions['boundary_position']*self.nely):int(self.conditions['boundary_position']*self.nely)+int(self.conditions['boundary_length']*self.nely),0]
            labels_load=[]
            for i in range(self.conditions['n_loads']):
            #labels of load:
                labels_load .append(labels[int(self.conditions['load_position'][i]*self.nely),-1])
        elif self.conditions['selected_boundary']==0.25: # Right boundary
            label_support = labels[int(self.conditions['boundary_position']*self.nely):int(self.conditions['boundary_position']*self.nely)+int(self.conditions['boundary_length']*self.nely),-1]
            labels_load=[]
            for i in range(self.conditions['n_loads']):
            #labels of load:
                labels_load .append(labels[int(self.conditions['load_position'][i]*self.nely),0])
        elif self.conditions['selected_boundary']==0.5: # Bottom boundary
            label_support = labels[-1,int(self.conditions['boundary_position']*self.nelx):int(self.conditions['boundary_position']*self.nelx)+int(self.conditions['boundary_length']*self.nelx)]
            labels_load=[]
            for i in range(self.conditions['n_loads']):
            #labels of load:
                labels_load .append(labels[0,int(self.conditions['load_position'][i]*self.nelx)])
        elif self.conditions['selected_boundary']==0.75: # Top boundary
            label_support = labels[0,int(self.conditions['boundary_position']*self.nelx):int(self.conditions['boundary_position']*self.nelx)+int(self.conditions['boundary_length']*self.nelx)]
            labels_load=[]
            for i in range(self.conditions['n_loads']):
            #labels of load:
                labels_load .append(labels[-1,int(self.conditions['load_position'][i]*self.nelx)])

        for load in labels_load:
             if load!=0:
                  if load in label_support:
                    connec.append(True)
                  else:
                    connec.append(False)
        # return True if connec is not empty and if all its elements are True
        return bool(connec) and all(connec)
    
