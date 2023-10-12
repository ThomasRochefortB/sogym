import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import numpy.matlib 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import numba



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


@numba.njit
def calc_Phi(variable, LSgrid, p):
    x0 = variable[0,:]
    y0 = variable[1,:]
    L = variable[2,:] + np.spacing(1)
    t1 = variable[3,:]
    t2 = variable[4,:]
    angle = variable[5,:]

    st = np.sin(angle)
    ct = np.cos(angle)

    x1 = ct*(LSgrid[0][:,None]-x0) + st*(LSgrid[1][:,None]-y0) + np.spacing(1)
    y1 = -st*(LSgrid[0][:,None]-x0) + ct*(LSgrid[1][:,None]-y0) + np.spacing(1)
    l = (t1+t2)/2 + (t2-t1)/2/L*x1 + np.spacing(1)
    temp = ((x1)**p)/((L**p)+1e-08) + ((y1)**p)/((l**p)+1e-08)
    allPhi = 1 - temp**(1/p)
    
    return allPhi

#Smoothed Heaviside function
def Heaviside(phi,alpha,epsilon):
    H = np.select([phi>epsilon, phi<-epsilon, True], [1, alpha, 3*(1-alpha)/4*(phi/epsilon-phi**3/(3*(epsilon)**3)) + (1+alpha)/2])
    return H
        
    
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
    loaddof_x=conditions['loaddof_x']
    loaddof_y=conditions['loaddof_y']
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
    
    freeDof = np.setdiff1d(np.arange(nDof),fixDof)         # index of free dofs
    try:
        F_x=csc_matrix((magnitude_x, (loaddof_x, np.zeros_like(loaddof_x))), shape=(nDof, 1))
    except:
        print('error while constructing F_x: ' , magnitude_x,loaddof_x)
    F_y=csc_matrix((magnitude_y, (loaddof_y, np.zeros_like(loaddof_y))), shape=(nDof, 1))
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
    
    return Comp[0][0],volume, U , F


#Function that takes a variable vector, and generates the contours of the components
def build_design(variable,DW=2.0,DH=1.0,nelx=100,nely=50):
    lmd = 100 
    nEle = nelx*nely             
    p=6
    alpha=1e-9 
    epsilon=0.2
    N=variable.shape[1]
    nNod = (nelx+1)*(nely+1)
    allPhi = np.zeros((nNod,N))   
    x,y=np.meshgrid(np.linspace(0, DW,nelx+1),np.linspace(0,DH,nely+1))
    LSgrid=np.array([x.flatten(order='F'),y.flatten(order='F')])
    nodMat = np.reshape(np.array(range(0,nNod)),(1+nely,1+nelx),order='F')                    
    edofVec = np.reshape(2*nodMat[0:-1,0:-1],(nEle,1),order='F')
    edofMat = edofVec + np.array([0, 1, 2*nely+2,2*nely+3, 2*nely+4, 2*nely+5, 2, 3])             
    eleNodesID = edofMat[:,0:8:2]//2   
    
    allPhi = calc_Phi(variable, LSgrid, p)
    temp = np.exp(lmd*allPhi)
    temp = np.where(temp==0,1e-08,temp)
    Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        
    
    H = Heaviside(Phimax,alpha,epsilon)                          
    den = np.sum(H[eleNodesID.astype('int')],1)/4                                 

    return H, Phimax,allPhi, den


