import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import math
import numpy.matlib
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags


# Element stiffness matrix (plane stress quadrilateral elements)
def Ke_tril(E,nu,a,b,h):
    """
    This function returns the stiffness matrix of a plane stress quadrilateral uniform/standard element
    """
    k1 = [-1/6/a/b*(nu*a**2-2*b**2-a**2),  1/8*nu+1/8, -1/12/a/b*(nu*a**2+4*b**2-a**2), 3/8*nu-1/8, 1/12/a/b*(nu*a**2-2*b**2-a**2),-1/8*nu-1/8,  1/6/a/b*(nu*a**2+b**2-a**2),   -3/8*nu+1/8]
    k2 = [-1/6/a/b*(nu*b**2-2*a**2-b**2),  1/8*nu+1/8, -1/12/a/b*(nu*b**2+4*a**2-b**2), 3/8*nu-1/8, 1/12/a/b*(nu*b**2-2*a**2-b**2),-1/8*nu-1/8,  1/6/a/b*(nu*b**2+a**2-b**2),   -3/8*nu+1/8]
    Ke = E*h/(1-nu**2)*np.array(
        [k1[0],k1[1],k1[2],k1[3],k1[4],k1[5],k1[6],k1[7],k2[0],k2[7],k2[6],k2[5],
         k2[4],k2[3],k2[2],k1[0],k1[5],k1[6],k1[3],k1[4],k1[1],k2[0],k2[7],k2[2],
         k2[1],k2[4],k1[0],k1[1],k1[2],k1[3],k2[0],k2[7],k2[6],k1[0],k1[5],k2[0]]
                                )
    return Ke


# Topology description function and derivatives
def calc_Phi(allPhi,xval,i,LSgrid,p,nEhcp,epsilon):
    
    dd = xval[:,i]
    x0 = dd[0]
    y0 = dd[1]
    L = dd[2] + np.spacing(1)
    t1 = dd[3]
    t2 = dd[4] 
    st = np.sin(dd[5])
    ct = np.cos(dd[5])  
    
    x1 = ct*(LSgrid['x']-x0) + st*(LSgrid['y']-y0) + np.spacing(1)                   # local x of each grid
    y1 = -st*(LSgrid['x']-x0) + ct*(LSgrid['y']-y0) + np.spacing(1)                  # local y of each grid
    l = (t1+t2)/2 + (t2-t1)/2/L*x1 + np.spacing(1)
    temp = ((x1)**p)/((L**p)+1e-08) + ((y1)**p)/((l**p)+1e-08)
    allPhi[:,i] = 1 - temp**(1/p)                                    # TDF of i-th component
    return allPhi

#Smoothed Heaviside function
def Heaviside(phi,alpha,epsilon):
    H = 3*(1-alpha)/4*(phi/epsilon-phi**3/(3*(epsilon)**3)) + (1+alpha)/2
    H=np.where(phi>epsilon,1,H)
    H=np.where(phi<-epsilon,alpha,H)
    return H

#Function that generates problem conditions (loading, support, volfrac)
def generate_problem(nelx=100,nely=50, mode = 'train'):
    """
    This function generates the problem conditions for the agent to solve
    """
    if mode == 'train':   # We define the training distribution of boundary conditions that the agent will encounter during training
        volfractions =[
            0.3,
            0.4,
            0.5
        ]
        conditions=['A','B','C','A_0','B_0','C_0']
        BC_saved=[]
        for volumes in volfractions:
            for cond in conditions:
                fixed_nodes=np.zeros(((nely+1),(nelx+1)))
                load_nodes=np.zeros(((nely+1),(nelx+1)))

                if cond == 'A': 
                    fixed_nodes[0,0:nelx//2]=1
                    load_nodes[:,-1]=1
                    direct=[45,90,225,270]

                if cond == 'B':
                    fixed_nodes[-1,0:nelx//2]=1
                    load_nodes[:,-1]=1
                    direct=[90,135,270,305]

                if cond == 'C':
                    fixed_nodes[:,0]=1
                    load_nodes[:,-1]=1
                    direct=[45,90,135,225,270,305]

                if cond =='A_0':
                    fixed_nodes[0,nelx//2+1:-1]=1
                    load_nodes[:,0]=1
                    direct=[90,135,270,305]

                if cond =='B_0':
                    fixed_nodes[-1,nelx//2+1:-1]=1
                    load_nodes[:,0]=1
                    direct=[45,90,225,270]

                if cond =='C_0':
                    fixed_nodes[:,-1]=1
                    load_nodes[:,0]=1
                    direct=[45,90,135,225,270,305]
                for angle in direct:
                    #generate the fixeddofs:
                    fixednode=np.argwhere(fixed_nodes.reshape((((nely+1))*((nelx+1))),order='F')==1)
                    fixeddofs=[]
                    for i in range(0,len(fixednode)):
                        fixeddofs.append(2*fixednode[i])
                        fixeddofs.append((2*fixednode[i])+1)

                    # generate the loaddofs:
                    loadnodes=np.argwhere(load_nodes.reshape((((nely+1))*((nelx+1))),order='F')==1)
                    for loadnode in loadnodes:
                        loaddof_x=[2*loadnode]
                        loaddof_y=[(2*loadnode)+1]

                        magnitude_x=np.array([np.cos(np.radians(angle))])
                        magnitude_y=np.array([np.sin(np.radians(angle))])        

                        data_save={
                            'fixeddofs':fixeddofs,
                            'loaddof_x':loaddof_x,
                            'loaddof_y':loaddof_y,
                            'magnitude_x':magnitude_x,
                            'magnitude_y':magnitude_y,
                            'volfrac':volumes,
                            'direct':direct,
                            'loadnode':loadnode,
                            'fixednode':fixednode
                        }
                        BC_saved.append(data_save)
        return BC_saved
                
    elif mode == 'test': # we define the test distribution of boundary conditions that the agent will encounter during testing to evaluate extrapolation generalization
        volfractions =[
            0.35,
            0.45,
            0.55
        ]
        conditions=['Y','Y_0','Z','Z_0']
        BC_saved=[]
        for volumes in volfractions:
            for cond in conditions:
                fixed_nodes=np.zeros(((nely+1),(nelx+1)))
                load_nodes=np.zeros(((nely+1),(nelx+1)))

                if cond == 'Y': 
                    fixed_nodes[0,nelx//4:-nelx//4]=1
                    load_nodes[-1,:]=1
                    direct=[0,180]

                if cond == 'Y_0':
                    fixed_nodes[-1,nelx//4:-nelx//4]=1
                    load_nodes[0,:]=1
                    direct=[0,180]

                if cond == 'Z':
                    fixed_nodes[nely//2:nely,0]=1
                    load_nodes[0,nelx//2:-1]=1
                    direct=[90,135,270,305]

                if cond =='Z_0':
                    fixed_nodes[nely//2:nely,-1]=1
                    load_nodes[0,0:nelx//2]=1
                    direct=[90,135,270,305]
                for angle in direct:
                    #generate the fixeddofs:
                    fixednode=np.argwhere(fixed_nodes.reshape((((nely+1))*((nelx+1))),order='F')==1)
                    fixeddofs=[]
                    for i in range(0,len(fixednode)):
                        fixeddofs.append(2*fixednode[i])
                        fixeddofs.append((2*fixednode[i])+1)

                    # generate the loaddofs:
                    loadnodes=np.argwhere(load_nodes.reshape((((nely+1))*((nelx+1))),order='F')==1)
                    for loadnode in loadnodes:
                        loaddof_x=[2*loadnode]
                        loaddof_y=[(2*loadnode)+1]

                        magnitude_x=np.array([np.cos(np.radians(angle))])
                        magnitude_y=np.array([np.sin(np.radians(angle))])        

                        data_save={
                            'fixeddofs':fixeddofs,
                            'loaddof_x':loaddof_x,
                            'loaddof_y':loaddof_y,
                            'magnitude_x':magnitude_x,
                            'magnitude_y':magnitude_y,
                            'volfrac':volumes,
                            'direct':direct,
                            'loadnode':loadnode,
                            'fixednode':fixednode
                        }
                        BC_saved.append(data_save)
        return BC_saved


    else:
        raise ValueError('mode must be either train or test')


    
#Function that takes the heaviside projection and the boundary conditions and calculates the compliance plus volume
def calculate_compliance(H,conditions,DW=2.0,DH=1.0,nelx=100,nely=50):
    """"
    This function takes the heaviside projection and the boundary conditions and calculates the compliance plus volume
    """
   
     # Material properties
    h=1 #thickness
    E=1
    nu=0.3
    
    # FEM data initialization
    EW=DW / nelx # length of element
    EH=DH / nely # width of element
    fixDof=conditions['fixeddofs']
    loaddof_x=conditions['loaddof_x'][0]
    loaddof_y=conditions['loaddof_y'][0]
    magnitude_x=conditions['magnitude_x']  
    magnitude_y=conditions['magnitude_y']
    
    ## Setting of FE discretization
    nEle = nelx*nely              # number of finite elements
    nNod = (nelx+1)*(nely+1)      # number of nodes
    nDof = 2*(nelx+1)*(nely+1)    # number of degree of freedoms
    Ke = Ke_tril(E,nu,EW,EH,h)  # non-zero upper triangular of ele. stiffness 
    KE=np.tril(np.ones(8)).flatten(order='F')
    KE[KE==1] = Ke.T
    KE=KE.reshape((8,8),order='F')
    KE = KE + KE.T - np.diag(np.diag(KE))           # full elemental stiffness matrix
    nodMat = np.reshape(np.array(range(0,nNod)),(1+nely,1+nelx),order='F')                    # maxtrix of nodes numbers (int32)
    edofVec = np.reshape(2*nodMat[0:-1,0:-1],(nEle,1),order='F')
    edofMat = edofVec + np.array([0, 1, 2*nely+2,2*nely+3, 2*nely+4, 2*nely+5, 2, 3])              # connectivity matrix
    eleNodesID = edofMat[:,0:8:2]/2    
    sI,sII = np.array([]),np.array([])
    for j in range(0,7):
        sI=np.concatenate((sI,np.linspace(j,7,8-j)))
        sII=np.concatenate((sII,np.matlib.repmat(j,1,8-j).squeeze()))
    sII=np.concatenate((sII,np.array([7])))
    sI=np.concatenate((sI,np.linspace(7,7,8-7)))
    iK,jK = edofMat[:,np.int32(sI)].T,edofMat[:,np.int32(sII)].T
    Iar0 = np.fliplr(np.sort(np.array([iK.flatten(order='F'),jK[:].flatten(order='F')]).T,axis=1) )        # reduced assembly indexing
    
    fixEle=[]
    for i in range(0,len(fixDof)):
        fixEle.append(np.argwhere(edofMat==fixDof[i])[0][0])                              # elements related to fixed nodes
    fixEle=np.unique(fixEle)
    freeDof = np.setdiff1d(np.arange(nDof),fixDof)         # index of free dofs

    loadEle=np.array([
        np.argwhere(edofMat==loaddof_x)[0][0],
        np.argwhere(edofMat==loaddof_y)[0][0]
    ])

    loadEle=np.unique(loadEle)
    F_x=csc_matrix(([magnitude_x[0]], ([loaddof_x[0]], [0])), shape=(nDof, 1))
    F_y=csc_matrix(([magnitude_y[0]], ([loaddof_y[0]], [0])), shape=(nDof, 1))
    F=F_x+F_y
    
    
   
    den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)
    U = np.zeros((nDof,1))
    
    sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(den,(den.shape[0],1)).T)
    sK = sK.flatten(order='F')
    K = csc_matrix((sK.flatten(order='F'), (Iar0[:,0], Iar0[:,1])), shape=(nDof, nDof))
    K =  K + K.T - diags((K.diagonal()))
    U[freeDof] =spsolve(K[freeDof,:][:,freeDof], F[freeDof]).reshape((len(freeDof),1))

    f0val = F.T*U
    
    Comp=f0val
    volume=sum(den)*EH*EW/(DH*DW)
    
    return Comp[0][0],volume


#Function that takes a variable vector, and generates the contours of the components
def build_design(variable,DW=2.0,DH=1.0,nelx=100,nely=50):
    xval=variable
    lmd = 100    #power of KS aggregation                                     

    nelx=100
    nely=50
    nEle = nelx*nely              # number of finite elements

    p=6
    alpha=1e-9 # parameter alpha in the Heaviside function
    epsilon=0.2
    N=variable.shape[1]
    actComp = np.arange(0,N)               # initial set of active components 
    nEhcp = 6                      # number design variables each component
    nNod = (nelx+1)*(nely+1)      # number of nodes
    
    allPhi = np.zeros((nNod,N))   
    x,y=np.meshgrid(np.linspace(0, DW,nelx+1),np.linspace(0,DH,nely+1))
    LSgrid={"x":x.flatten(order='F'),"y":y.flatten(order='F')}
    nodMat = np.reshape(np.array(range(0,nNod)),(1+nely,1+nelx),order='F')                    # maxtrix of nodes numbers (int32)
    edofVec = np.reshape(2*nodMat[0:-1,0:-1],(nEle,1),order='F')
    edofMat = edofVec + np.array([0, 1, 2*nely+2,2*nely+3, 2*nely+4, 2*nely+5, 2, 3])              # connectivity matrix
    eleNodesID = edofMat[:,0:8:2]/2    
    
    for i in actComp:                      # calculating TDF of the active MMCs                                                            
        allPhi = calc_Phi(allPhi,xval,i,LSgrid,p,nEhcp,epsilon)
    temp = np.exp(lmd*allPhi)
    temp = np.where(temp==0,1e-08,temp)
    Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        # global TDF using K-S aggregation
     #%--------------------------LP 3): Finite element analysis
    H = Heaviside(Phimax,alpha,epsilon)                            # nodal density vector 
    den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)

    return H, Phimax,allPhi, den