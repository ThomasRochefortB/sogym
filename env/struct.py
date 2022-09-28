import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import math
import numpy.matlib
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


#Element stiffness matrix
def BasicKe(E,nu, a, b,h):
    k=np.array([-1/6/a/b*(nu*a**2-2*b**2-a**2), 1/8*nu+1/8, -1/12/a/b*(nu*a**2+4*b**2-a**2),3/8*nu-1/8, 1/12/a/b*(nu*a**2-2*b**2-a**2),-1/8*nu-1/8, 1/6/a/b*(nu*a**2+b**2-a**2), -3/8*nu+1/8])
    KE=E*h/(1-nu**2)*np.array(
    [ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    return KE


#Forming Phi_i for each component
def tPhi(xy,LSgridx,LSgridy,p):
    st=xy[6]
    ct=np.sqrt(abs(1-(st*st)))
    x1=ct*(LSgridx - xy[0])+st*(LSgridy - xy[1])
    y1=-st*(LSgridx - xy[0])+ct*(LSgridy -xy[1])
    bb=((xy[4]+xy[3]-2*xy[5])/((2*xy[2]**2)))*(x1**2) + ((xy[4]-xy[3])/((2*xy[2])))*x1 + xy[5]
    tmpPhi=-(((x1)**p)/((xy[2]**p)) + (((y1)**p)/((bb**p))) -1)
    return tmpPhi

#Heaviside function
def Heaviside(phi,alpha,nelx,nely,epsilon):
    phi=phi.flatten(order='F')
    num_all=np.arange(0,(nelx+1)*(nely+1))
    H=np.ones((nelx+1)*(nely+1))
    H=np.where(phi<-epsilon,alpha,H)
    H=np.where((phi>= -epsilon) & (phi<= epsilon),((3*(1-alpha)/4)*((phi/epsilon)-(phi**3/(3*(epsilon)**3)))+(1+alpha)/2),H)
    return H

#Function that generates problem conditions (loading, support, volfrac)
def generate_problem():
    nelx=100
    nely=50
    #Changing conditions
    volfractions=[0.3,0.4,0.5]
    conditions=['A','B','C','A_0','B_0','C_0']
    BC_saved=[]
    for volumes in volfractions:
        for cond in conditions:
            fixed_nodes=np.zeros(((nely+1),(nelx+1)))
            load_nodes=np.zeros(((nely+1),(nelx+1)))

            if cond == 'A': 
                fixed_nodes[0,0:50]=1
                load_nodes[:,-50:]=1
                direct=[45,90,225,270]

            if cond == 'B':
                fixed_nodes[-1,0:50]=1
                load_nodes[:,-50:]=1
                direct=[90,135,270,305]

            if cond == 'C':
                fixed_nodes[:,0]=1
                load_nodes[:,-50:]=1
                direct=[45,90,135,225,270,305]

            if cond =='A_0':
                fixed_nodes[0,51:-1]=1
                load_nodes[:,0:50]=1
                direct=[90,135,270,305]

            if cond =='B_0':
                fixed_nodes[-1,51:-1]=1
                load_nodes[:,0:50]=1
                direct=[45,90,225,270]

            if cond =='C_0':
                fixed_nodes[:,-1]=1
                load_nodes[:,0:50]=1
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

#Function that takes the heaviside projection and the boundary conditions and calculates the compliance plus volume
def calculate_compliance(H,conditions):
    
    DW=2.0
    DH=1.0
    nelx=100
    nely=50
     # Material properties
    h=1 #thickness
    E=1
    nu=0.3
    
    # FEM data initialization
    M=[nely+1, nelx+1]
    EW=DW / nelx # length of element
    EH=DH / nely # width of element
    fixeddofs=conditions['fixeddofs']
    loaddof_x=conditions['loaddof_x'][0]
    loaddof_y=conditions['loaddof_y'][0]
    magnitude_x=conditions['magnitude_x']  
    magnitude_y=conditions['magnitude_y']
    
    #Define loads and supports(Short beam)
     #Define loads and supports(Short beam)
    alldofs=np.arange(0,2*(nely+1)*(nelx+1))
    freedofs=np.setdiff1d(alldofs,fixeddofs)
    F_x=csc_matrix(([magnitude_x[0]], ([loaddof_x[0]], [0])), shape=(2*(nely+1)*(nelx+1), 1))
    F_y=csc_matrix(([magnitude_y[0]], ([loaddof_y[0]], [0])), shape=(2*(nely+1)*(nelx+1), 1))
    F=F_x+F_y
    
    nodenrs=np.arange(0,(1+nelx)*(1+nely)).reshape(1+nely,1+nelx,order='F')
    edofVec=((2*nodenrs[0:-1,0:-1])).reshape(nelx*nely,1,order='F')
    edofMat=np.matlib.repmat(edofVec,1,8)+np.matlib.repmat(np.concatenate([np.array([0,1]),2*nely+np.array([2,3,4,5]),np.array([2,3])],axis=0),nelx*nely,1)
    iK=np.kron(edofMat,np.ones((8,1))).T
    jK=np.kron(edofMat,np.ones((1,8))).T
    EleNodesID=(edofMat[:,(1,3,5,7)]-1)/2
    iEner=EleNodesID.T
    KE=BasicKe(E,nu, EW, EH,h) # stiffness matrix k**s is formed
    #FEA
    denk=np.sum(H[EleNodesID.astype(int)]**2, axis=1) / 4
    den=np.sum(H[EleNodesID.astype(int)], axis=1) / 4
    A1=np.sum(den)*EW*EH
    U=np.zeros((2*(nely+1)*(nelx+1),1))
    sK=np.expand_dims(KE.flatten(order='F'),axis=1)@(np.expand_dims(denk.flatten(order='F'),axis=1).T)
    K = coo_matrix((sK.flatten(order='F'),(iK.flatten(order='F'),jK.flatten(order='F'))),shape=(2*(nely+1)*(nelx+1),2*(nely+1)*(nelx+1))).tocsc()
    # Remove constrained dofs from matrix
    K = K[freedofs,:][:,freedofs]
    # Solve system 
    U[freedofs,0]=spsolve(K,F[freedofs,0])
    #Energy of element
    energy=np.sum((U[edofMat][:,:,0]@KE)*U[edofMat][:,:,0],axis=1)
    sEner=np.ones((4,1))@np.expand_dims(energy,axis=1).T/4
    energy_nod=csc_matrix((sEner.flatten(order='F'), (iEner.flatten(order='F').astype(int),np.zeros((len(sEner.flatten(order='F')),)).astype(int))))
    Comp=F.T@U
    volume=A1/(DW*DH)
    
    #Initialize empty matrices
    u_x=np.zeros(((nely+1),((nelx+1))))
    u_y=np.zeros(((nely+1),((nelx+1))))

    counterx=0
    for j in range(0,np.shape(u_x)[1]):
        for i in range(0,np.shape(u_x)[0]):
            u_x[i][j]=U[counterx]
            counterx+=2

    countery=1
    for j in range(0,np.shape(u_y)[1]):
        for i in range(0,np.shape(u_y)[0]):
            u_y[i][j]=U[countery]
            countery+=2
    Umag=np.sqrt(u_x**2+u_y**2)
    #Creation of B, the strain-displacement matrix. Here it is precomputed for a regular 2D square element
    n = -0.5
    nu=0.3
    p = 0.5 
    B=np.array([[n,0,p,0,p,0,n,0],
                [0,n,0,n,0,p,0,p],
                [n,n,n,p,p,p,p,n]])
    Emax=1.0
    Emin=1e-6
    #Creation of E,
    E_filled=np.array([[1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2.]])*(Emax / (1 - nu**2))

    E_void=np.array([[1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2.]])*(Emin / (1 - nu**2))

    # Creation of EB, dot product of E*B
    EB_filled = E_filled.dot(B)
    EB_void=E_void.dot(B)
    # Now we need to iterate through our displacements, and gather all the displacements of the 4 nodes for each element

    #Initialize some empty matrices:
    stressx=[]
    stressy=[]
    shear=[]
    vmstress=[]

    ### Will output a list with each element's sigmax,sigmay and sigmaxy (shear)
    elem_disp=np.zeros((8,1))

    for i in range(0,len(edofMat)):
        elem_disp[0,0]=U[edofMat[i][0]]
        elem_disp[1,0]=U[edofMat[i][1]]
        elem_disp[2,0]=U[edofMat[i][2]]
        elem_disp[3,0]=U[edofMat[i][3]]
        elem_disp[4,0]=U[edofMat[i][4]]
        elem_disp[5,0]=U[edofMat[i][5]]
        elem_disp[6,0]=U[edofMat[i][6]]
        elem_disp[7,0]=U[edofMat[i][7]]

        #Principal stress (dot product of EB with element displacement)

        if denk[i]<0.5:
            s11, s22, s12 = EB_void.dot(elem_disp)
        else:
            s11, s22, s12 = EB_filled.dot(elem_disp)
        #s11, s22, s12 = EB_filled.dot(elem_disp)

        #Von mises stress
        vm_stress = np.sqrt((s11**2) - (s11 * s22) + (s22**2) + (3 * s12**2))  #General plane stress
        vmstress.append(vm_stress)
    vmstress=np.array(vmstress).reshape((nely,nelx),order='F')
    return Comp,volume,vmstress


#Function that takes a variable vector, and generates the contours of the components
def build_design(variable):
    DW=2.0
    DH=1.0
    nelx=100
    nely=50
    p=6
    alpha=1e-9 # parameter alpha in the Heaviside function
    epsilon=0.2

    Var_num=7 # number of design variablesfor each component


    M=[nely+1, nelx+1]
    EW=DW / nelx # length of element
    EH=DH / nely # width of element
    x,y=np.meshgrid(np.linspace(0, DW,nelx+1),np.linspace(0,DH,nely+1))
    LSgrid={"x":x.flatten(order='F'),"y":y.flatten(order='F')}
    
    #We need to take into account a variable number of components
    N=variable.shape[1]
    Phi=[None] * N
    #Forming Phi^s
    for i in range(0,N):
        Phi[i]=tPhi(variable[:,i],LSgrid['x'],LSgrid['y'],p)
    Phi_out=Phi
    #Union of components
    tempPhi_max=Phi[0]

    for i in range(1,N):
        tempPhi_max=np.maximum(tempPhi_max,Phi[i])
    
    Phi_max=tempPhi_max.reshape((nely+1,nelx+1),order='F')
    #Heaviside projection:
    H=Heaviside(Phi_max,alpha,nelx,nely,epsilon)
    
    return H, Phi_out