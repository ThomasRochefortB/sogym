import gym
from env.struct import *
import numpy as np
import random
import matplotlib.pyplot as plt

#Class defining the Structural Optimization Gym environment (so-gym):
class sogym(gym.Env):

    def __init__(self):
        self.nelx=100
        self.nely=50
        self.DW=2.0
        self.DH=1.0
        self.EW=self.DW / self.nelx # length of element
        self.EH=self.DH / self.nely # width of element
        self.N_components=8
        self.xmin=np.vstack((0, 0, 0.03, 0.03, 0.03, 0.03, -1.0))
        self.xmax=np.vstack((self.DW, self.DH, 1.0, 0.1, 0.1, 0.1, 1.0))
        self.BC_dict=generate_problem()
        self.x,self.y=np.meshgrid(np.linspace(0, self.DW,self.nelx+1),np.linspace(0,self.DH,self.nely+1))                # coordinates of nodal points

        
        self.render_colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']

        self.action_space = gym.spaces.Box(low=-1,high=1,shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
                                    spaces={
                                        "conditions": gym.spaces.Box(-1, 1, (9,),dtype=np.float64),
                                        "n_steps_left":gym.spaces.Discrete(self.N_components),
                                        "design_variables": gym.spaces.Box(-1, 2, (self.N_components*7,),dtype=np.float64),
                                        "volume":gym.spaces.Box(0,1,(1,),dtype=np.float64),
                                        }
                                    )    

    def reset(self):
        self.variables_plot=[]
        self.conditions=self.BC_dict[random.randint(0,len(self.BC_dict)-1)]
        fixednodes=self.conditions['fixednode']
        loadnode=self.conditions['loadnode']
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
        features=np.array([load_coords[0]/(self.nely+1),  #Relative y position of load
                                  load_coords[1]/(self.nelx+1),  #Relative x position of load
                                  fixed_coord1[0]/(self.nely+1), #Relative y position of support_1
                                fixed_coord1[1]/(self.nelx+1),   #Relative x position of support_1
                                  fixed_coord2[0]/(self.nely+1), #Relative y position of support_2
                                  fixed_coord2[1]/(self.nelx+1), #Relative x position of support_2
                                  volfrac,                  #Volume fraction (between 0 and 1)
                                  magnitude_x,              #magnitude of load in x 
                                  magnitude_y  ])          #magnitude of load in y 
        
        self.variables=np.zeros((8*7,1))
        self.out_conditions=features
        self.action_count=0
        self.saved_volume=[0.0]
        self.observation={"conditions":self.out_conditions,
                          "design_variables":self.variables.flatten(),
                         "volume":np.array([0.0]),
                         "n_steps_left":self.N_components-self.action_count-1,

                            }


        return self.observation 
        
        
    def step(self, action):
        self.action_count+=1
       
        new_variables=(self.xmax.squeeze()-self.xmin.squeeze())/(2)*(action-1)+self.xmax.squeeze()
        self.variables[(self.action_count-1)*7:self.action_count*7,0]=new_variables
        
        
        self.variables_plot.append(new_variables)
        self.H, self.Phi=build_design(np.array(self.variables_plot).T)
        self.proj_img=self.H.reshape((1,self.nely+1,self.nelx+1),order='F')
        
        nodenrs=np.arange(0,(1+self.nelx)*(1+self.nely)).reshape(1+self.nely,1+self.nelx,order='F')
        edofVec=((2*nodenrs[0:-1,0:-1])).reshape(self.nelx*self.nely,1,order='F')
        edofMat=np.matlib.repmat(edofVec,1,8)+np.matlib.repmat(np.concatenate([np.array([0,1]),2*self.nely+np.array([2,3,4,5]),np.array([2,3])],axis=0),self.nelx*self.nely,1)
        EleNodesID=(edofMat[:,(1,3,5,7)]-1)/2
        #FEA
        self.den=np.sum(self.H[EleNodesID.astype(int)], axis=1) / 4
        self.volume=np.sum(self.den)/(self.nelx*self.nely)
        
        if self.action_count<self.N_components:
            reward=0
            done=False
            
        else:
            
            compliance,self.volume,self.vmstress=calculate_compliance(self.H,self.conditions)
            
            if self.volume<= self.out_conditions[6]:
                reward=(1/(compliance+1e-8))
                reward=reward[0][0]
                self.final_img=self.proj_img
            else:
                reward=0
                
            done=True
            
        info={}
        obs_dict={"conditions":self.out_conditions,
                 "design_variables":self.variables.flatten(),
                 "volume":np.array([self.volume]),
                 "n_steps_left":self.N_components-self.action_count-1
                 }
        self.saved_volume.append(self.volume)
        self.observation=obs_dict
        self.plot_conditions = self.out_conditions

        return self.observation, reward, done, info
    

    def plot(self, mode='human',test=None):
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
        for i, color in zip(range(0,len(self.Phi)), self.render_colors):
            ax.contourf(self.x,self.y,np.flipud(self.Phi[i].reshape((self.nely+1,self.nelx+1),order='F')),[0,1],colors=color)
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
        plt.close()
        return fig    
        
