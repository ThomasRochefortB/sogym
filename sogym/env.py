import gym
from sogym.struct import *
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

#Class defining the Structural Optimization Gym environment (so-gym):
class sogym(gym.Env):

    def __init__(self,N_components=8,nelx=100,nely=50,DW=2.0,DH=1.0,observation_type = 'dense',mode = 'train'):
        self.nelx = nelx
        self.nely = nely
        self.DW = DW
        self.DH = DH
        self.EW=self.DW / self.nelx # length of element
        self.EH=self.DH / self.nely # width of element
        self.N_components = N_components
        self.mode = mode
        self.observation_type = observation_type
        # Agent's actions control the (x,y) coordinates of the two endpoints as well as two thicknesses
        self.xmin=np.vstack((0, 0, 0.0, 0.0, 0.0, 0.0))  # (xa_min,ya_min, xb_min, yb_min, t1_min, t2_min)
        self.xmax=np.vstack((self.DW, self.DH, self.DW, self.DH, 0.3, 0.3)) # (xa_max,ya_max, xb_max, yb_max, t1_max, t2_max)
        self.N_actions = 6
        
        self.BC_dict=generate_problem(nelx,nely,mode)
        self.x,self.y=np.meshgrid(np.linspace(0, self.DW,self.nelx+1),np.linspace(0,self.DH,self.nely+1))                # coordinates of nodal points

        # series of render color for the plot function
        self.render_colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']

        self.action_space = gym.spaces.Box(low=-1,high=1,shape=(self.N_actions,), dtype=np.float32)
        
        if self.observation_type =='dense':
            self.observation_space = gym.spaces.Dict(
                                        spaces={
                                            "conditions": gym.spaces.Box(-1, 1, (9,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":gym.spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": gym.spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":gym.spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )
        
        elif self.observation_type =='box_dense':
                                         self.observation_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(9+1+1+self.N_components*self.N_actions,), dtype=np.float32) 

        elif self.observation_type =='image':
            #To define the image space here.
            self.observation_space = gym.spaces.Dict(
                                        spaces={
                                            "image": gym.spaces.Box(0, 255, (3,64,128),dtype=np.uint8), # Image of the current design
                                            #"image": gym.spaces.Box(0, 255, (64,128,3),dtype=np.uint8), # Image of the current design
                                            "conditions": gym.spaces.Box(-1, 1, (9,),dtype=np.float32), # Description vector \beta containing (TO DO)
                                            "n_steps_left":gym.spaces.Box(0.0,1.0,(1,),dtype=np.float32),
                                            "design_variables": gym.spaces.Box(-1.0, 1.0, (self.N_components*self.N_actions,),dtype=np.float32),
                                            "volume":gym.spaces.Box(0,1,(1,),dtype=np.float32), # Current volume at the current step
                                            }
                                        )   
            pass
        else:
            raise ValueError('Invalid observation space type. Only "dense", "box_dense" and "image" are supported.')

    def reset(self):
        self.random_int = random.randint(0,len(self.BC_dict)-1)
        self.variables_plot=[]
        self.conditions=self.BC_dict[self.random_int]
        load_mat_print=np.zeros(((self.nely+1)*(self.nelx+1),1))
        load_mat_print[self.conditions['fixednode']]=1
        load_mat_print[self.conditions['loadnode'],0]=2
        load_mat_print=load_mat_print.reshape((self.nely+1,self.nelx+1,1),order='F')

        load_coords=np.argwhere(load_mat_print==2)[0][0:2]
        fixed_coord1=np.argwhere(load_mat_print==1)[0][0:2]
        fixed_coord2=np.argwhere(load_mat_print==1)[-1][0:2]
        volfrac=np.array(self.conditions['volfrac'])
        magnitude_x=self.conditions['magnitude_x'][0]
        magnitude_y=self.conditions['magnitude_y'][0]


        beta=np.array([load_coords[0]/(self.nely+1),  #Relative y position of load
                                  load_coords[1]/(self.nelx+1),  #Relative x position of load
                                  fixed_coord1[0]/(self.nely+1), #Relative y position of support_1
                                fixed_coord1[1]/(self.nelx+1),   #Relative x position of support_1
                                  fixed_coord2[0]/(self.nely+1), #Relative y position of support_2
                                  fixed_coord2[1]/(self.nelx+1), #Relative x position of support_2
                                  volfrac,                  #Volume fraction (between 0 and 1)
                                  magnitude_x,              #magnitude of load in x 
                                  magnitude_y  ])          #magnitude of load in y 
        
        self.variables=np.zeros((self.N_components*self.N_actions,1))
        self.out_conditions=beta
        self.action_count=0
        self.saved_volume=[0.0]
        self.plot_conditions = self.out_conditions
        # I need to initialize an empty instance of Phi:
        self.Phi = np.zeros(((self.nelx+1)*(self.nely+1), self.N_components))
        if self.observation_type=='dense':
            self.observation={"conditions":np.float32(self.out_conditions),
                            "design_variables":np.float32(self.variables.flatten()),
                            "volume":np.array([0.0],dtype=np.float32),
                            "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                            }
        elif self.observation_type=='box_dense':        
            self.observation=np.concatenate(
                (np.float32(self.out_conditions),
                 np.array([0.0],dtype=np.float32),
                 np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                 np.float32(self.variables.flatten()))
                 ,axis=0)

        elif self.observation_type=='image':
            self.observation={"image":self.gen_image(resolution=(128,64)),
                            "conditions":np.float32(self.out_conditions),
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
        self.H, self.Phimax,self.Phi, den=build_design(np.array(self.variables_plot).T, self.DW,self.DH, self.nelx,self.nely)    # self.H is the Heaviside projection of the design variables and self.Phi are the design surfaces.
        self.proj_img=self.H.reshape((1,self.nely+1,self.nelx+1),order='F')
        
        nEle = self.nelx*self.nely
        nNod=(self.nelx+1)*(self.nely+1)
        nodMat = np.reshape(np.array(range(0,nNod)),(1+self.nely,1+self.nelx),order='F')                    # maxtrix of nodes numbers (int32)
        edofVec = np.reshape(2*nodMat[0:-1,0:-1],(nEle,1),order='F')
        edofMat = edofVec + np.array([0, 1, 2*self.nely+2,2*self.nely+3, 2*self.nely+4, 2*self.nely+5, 2, 3])              # connectivity matrix
        eleNodesID = edofMat[:,0:8:2]/2   
        #FEA
        self.den=np.sum(self.H[eleNodesID.astype('int')],1)/4 
        self.volume=sum(self.den)*self.EW*self.EH/(self.DW*self.DH)
        

        # The reward function is the sparse reward given only at the end of the episode if the desired volume fraction is respected.
        if self.action_count<self.N_components: # We are not at the end of the episode
            reward=0.0
            done=False

        else: # We are at the end of the episode
            done=True
            self.last_Phi = self.Phi
            self.compliance,self.volume, self.U, self.F=calculate_compliance(self.H,self.conditions,self.DW,self.DH,self.nelx,self.nely) # We calculate the compliance, volume and von Mises stress of the structure
            
            if self.volume<= self.out_conditions[6] and self.check_connec(): # The desired volume fraction is respected
                reward=(1/(self.compliance+1e-8)) # The reward is the inverse of the compliance (AKA the stiffness of the structure)
            else:
                reward=0.0
            
        info={}
        if self.observation_type=='dense':
            self.observation = {"conditions":np.float32(self.out_conditions),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                    "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                    }
        elif self.observation_type=='box_dense':        
            self.observation=np.concatenate(
                (np.float32(self.out_conditions),
                 np.array([self.volume],dtype=np.float32),
                 np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                 np.float32(self.variables.flatten())/np.pi)
                 ,axis=0)
                 
        elif self.observation_type=='image':
            self.observation = {"image":self.gen_image(resolution=(128,64)),
                    "conditions":np.float32(self.out_conditions),
                    "design_variables":np.float32(self.variables.flatten())/np.pi,
                    "volume":np.array([self.volume],dtype=np.float32),
                    "n_steps_left":np.array([(self.N_components - self.action_count) / self.N_components],dtype=np.float32),
                    }
        else:
            raise ValueError('Invalid observation space type. Only "dense" and "image" are supported.')
        self.saved_volume.append(self.volume)

        return self.observation, reward, done, info
    

    def plot(self, mode='human',test=None, train_viz=True):
        plt.rcParams["figure.figsize"] = (10,5)
        X=self.plot_conditions
        
        if test is not None:
            X=test
            
        load_A1=X[0]*1
        load_A2=X[1]*2
        fixed_A1=X[2]
        fixed_A2=X[3]*2
        fixed_B1=X[4]
        fixed_B2=X[5]*2
        magx=X[7]
        magy=X[8]

        patch_x=fixed_A2
        patch_y=1-fixed_A1

        if fixed_A2==fixed_B2: #X coordinate
            if patch_x<0.1:
                patch_width=-0.1
            else:
                patch_width=0.1
        else:
            patch_width=fixed_B2-fixed_A2

        if fixed_A1==fixed_B1:# Y coordinate

            if patch_y<0.1:
                patch_height=-0.1
            else:
                patch_height=0.1

        else:
            patch_height=fixed_A1-fixed_B1

        if patch_height<-0.5:
            patch_height=-1

        if patch_width>0.5:
            patch_width=1


        fig = plt.figure()
        ax = plt.subplot(111)
        if train_viz:
            for i, color in zip(range(0,self.last_Phi.shape[1]), self.render_colors):
                ax.contourf(self.x,self.y,np.flipud(self.last_Phi[:,i].reshape((self.nely+1,self.nelx+1),order='F')),[0,1],colors=color)
        else:
            for i, color in zip(range(0,self.Phi.shape[1]), self.render_colors):
                ax.contourf(self.x,self.y,np.flipud(self.Phi[:,i].reshape((self.nely+1,self.nelx+1),order='F')),[0,1],colors=color)
        
        ax.add_patch(plt.Rectangle((patch_x+0.01,patch_y),patch_width, patch_height,hatch='/',
                                      clip_on=False,linewidth = 0))

        ax.add_patch(plt.Rectangle((0,0),2.0, 1.0,
                                      clip_on=False,linewidth = 1,fill=False))

        factor=0.2
        ax.arrow(load_A2-(magx*factor),(1-load_A1-(magy*factor)),
                      magx*factor,
                      magy*factor,
                      width=factor/8,
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
        res = np.moveaxis(res, -1, 0)
        return res

    def check_connec(self):

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
        y_load = int(self.out_conditions[0]*(self.nely))
        x_load = int(self.out_conditions[1]*(self.nelx))
        y_support_1 = int(self.out_conditions[2]*(self.nely))
        x_support_1 = int(self.out_conditions[3]*(self.nelx))
        y_support_2 = int(self.out_conditions[4]*(self.nely))
        x_support_2 = int(self.out_conditions[5]*(self.nelx))

        #labels of support_1 and support_2
        label_support_1 = labels[y_support_1,x_support_1:x_support_2]

        #labels of load:
        label_load = labels[y_load,x_load]
        if label_load !=0:
            if label_load in label_support_1:
                return True
        else:
            return False
