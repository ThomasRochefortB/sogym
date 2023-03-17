import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'       #Disactivate multiprocessing for numpy
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import matplotlib.pyplot as plt
from IPython import display
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sogym.mma import mmasub,gcmmasub,asymp,concheck,raaupdate
from sogym.struct import Ke_tril, Heaviside


#Loading path identification
def srch_ldpth(nAct,allPhiAct,Phimax,epsilon,eleNodesID,loadEle,fixEle):
    strct = 0
    fixSet = np.array([])
    loadPth = np.array([])
    Frnt = np.array([])
    # initialization
    Hmax = Heaviside(Phimax,0,epsilon)
    denmax = np.sum(Hmax[eleNodesID.astype('int')], 1)/4
    allH = Heaviside(allPhiAct,0,epsilon)
    allden = np.matlib.repmat(denmax.reshape((denmax.shape[0],1)),1,nAct)
    for i in range(0,nAct):  
        tempH = allH[:,i]
        allden[:,i] = np.sum(tempH[eleNodesID.astype('int')], 1)/4                  # density matrix of all active components 

    if min(denmax[loadEle.astype('int')[:]])>np.spacing(1) and max(denmax[fixEle[:]])>np.spacing(1):
        cnnt = csc_matrix((nAct,nAct))                   # connection matrix of components
        for i in range(0,nAct):
            for j in range(i+1,nAct):
                if max(np.min(allden[:,[i,j]],axis=1))>0:
                    cnnt[i,j] = 1
                    cnnt[j,i] = 1 
            if max(allden[loadEle.astype('int'),i])>0:
                loadPth = np.unique(np.append(loadPth, i))
                Frnt = np.unique(np.setdiff1d(np.append(Frnt, np.argwhere(cnnt[i,:]>0)), loadPth))
            if max(allden[fixEle,i])>0:
                fixSet = np.unique(np.append(fixSet, i))
        while len(Frnt)>0:
            loadPth = np.sort(np.append(loadPth,Frnt))
            Temp = []
            for i in Frnt:
                Temp = np.append(Temp, np.argwhere(cnnt[i,:]>0))
            Frnt = np.unique(np.setdiff1d(Temp,loadPth))

        if len(np.intersect1d(loadPth,fixSet))>0:
            strct = 1

    return [strct,loadPth]
# Topology description function and derivatives
def calc_Phi(allPhi,allPhidrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb):
    dd = xval[np.arange((i)*nEhcp,(i+1)*nEhcp)]
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
    temp = (x1)**p/L**p + (y1)**p/l**p
    allPhi[:,i] = 1 - temp**(1/p)                                    # TDF of i-th component
    #switched the threshold for  np.mean(dd[3:5])/minSz from 0.1 to 0.2
   # if np.mean(dd[3:5])/minSz > 0.5 and min(abs(allPhi[:,i])) < epsilon:
    dx1 = np.array([-ct+0.0*x1, -st+0.0*x1, 0.0*x1, 0.0*x1, 0.0*x1, y1])        # variation of x'
    dy1 = np.array([st+0.0*y1, -ct+0.0*y1, 0.0*y1, 0.0*y1, 0.0*y1, -x1])        # variation of y'
    dldx1 = (t2-t1)/2/L
    dldv1 = 0.0*l   #dldx0
    dldv2 = 0.0*l   #dldy0
    dldv6 = 0.0*l   #dldtheta               
    dldv3 = - (t2-t1)/2*x1/L**2
    dldv4 = 1/2 - x1/2/L
    dldv5 = 1/2 + x1/2/L
    dl = np.array([dldv1, dldv2, dldv3, dldv4, dldv5, dldv6]).T + np.multiply(np.matlib.repmat(dldx1,1,nEhcp),dx1.T);  # variation of width
    dpdx1 = -(temp)**(1/p-1)*((x1/L)**(p-1)/L)                     # dphi/dx'
    dpdx1=np.reshape(dpdx1,(dpdx1.shape[0],1))
    dpdy1 = -(temp)**(1/p-1)*((y1/l)**(p-1)/l)                    # dphi/dy'
    dpdy1 = np.reshape(dpdy1,(dpdy1.shape[0],1))
    dpdL = 0.0*dx1.T
    dpdL[:,2] = (temp)**(1/p-1)*(x1/L)**p/L    # [0 0 dphi/dl 0 0 0 0]         
    dpdl = (temp)**(1/p-1)*((y1/l)**p/l)
    dpdl = np.reshape(dpdl,(dpdl.shape[0],1))
    operation = np.multiply(np.matlib.repmat(dpdx1,1,nEhcp),dx1.T) + np.multiply(np.matlib.repmat(dpdy1,1,nEhcp),dy1.T) + dpdL + np.multiply(np.matlib.repmat(dpdl,1,nEhcp),dl)
    allPhidrv[:,np.arange((i)*nEhcp,(i+1)*nEhcp)] = operation
    return [allPhi,allPhidrv,xval,actComp,actDsvb]



def run_mmc(BC_dict,nelx=100,nely=50,DW=2.0,DH=1.0,plotting='contour'):   ## Probably need to add xmin and xmax
    xInt = 0.5
    yInt = 0.25
    vInt = [0.4, 0.05, 0.05, np.arcsin(0.7)]
    E = 1.0 #Young's modulus
    nu = 0.3 #Poisson ratio
    h = 1 #thickness                              
    dgt0 = 5 #significant digit of sens.
    scl = 1  #scale factor for obj                                           
    p = 6   #power of super ellipsoid
    lmd = 100    #power of KS aggregation                                     
    maxiter = 1000 # maximum number of iterations                                       
    objVr5 = 1.0  # initial relative variat. of obj. last 5 iterations

    ## Setting of FE discretization
    nEle = nelx*nely              # number of finite elements
    nNod = (nelx+1)*(nely+1)      # number of nodes
    nDof = 2*(nelx+1)*(nely+1)    # number of degree of freedoms
    EL = DW/nelx                  # length of finite elements
    EW = DH/nely                  # width of finite elements
    minSz = min([EL,EW])*3          # minimum size of finite elements
    alpha = 1e-9                  # void density
    epsilon = 0.2              # regularization term in Heaviside (default 0.2)
    Ke = Ke_tril(E,nu,EL,EW,h)  # non-zero upper triangular of ele. stiffness 
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
    x,y=np.meshgrid(np.linspace(0, DW,nelx+1),np.linspace(0,DH,nely+1))                # coordinates of nodal points
    LSgrid={"x":x.flatten(order='F'),"y":y.flatten(order='F')}
    volNod=csc_matrix((np.ones(eleNodesID.size)/4,(eleNodesID.flatten(order='F'),np.zeros(eleNodesID.size))),shape=(nNod,1))                       # weight of each node in volume calculation 

#  3): LOADS, DISPLACEMENT BOUNDARY CONDITIONS (2D cantilever beam example)
    volfrac=BC_dict['volfrac']
    magnitude_x=BC_dict['magnitude_x']
    magnitude_y=BC_dict['magnitude_y']
    loaddof_x=BC_dict['loaddof_x']                 # loaded dofs
    loaddof_y=BC_dict['loaddof_y']                 # loaded dofs
    fixDof=BC_dict['fixeddofs']      # fixed nodes
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
    F_x=csc_matrix(([magnitude_x[0]], ([loaddof_x[0][0]], [0])), shape=(nDof, 1))
    F_y=csc_matrix(([magnitude_y[0]], ([loaddof_y[0][0]], [0])), shape=(nDof, 1))
    F=F_x+F_y
    
    data=BC_dict
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
    X=np.array([load_coords[0]/(nely+1),  #Relative y position of load
                            load_coords[1]/(nelx+1),  #Relative x position of load
                            fixed_coord1[0]/(nely+1), #Relative y position of support_1
                            fixed_coord1[1]/(nelx+1),   #Relative x position of support_1
                            fixed_coord2[0]/(nely+1), #Relative y position of support_2
                            fixed_coord2[1]/(nelx+1), #Relative x position of support_2
                            volfrac,                  #Volume fraction (between 0 and 1)
                            magnitude_x,              #magnitude of load in x 
                            magnitude_y,              #magnitude of load in y 
                            ])   

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

    #  4): INITIAL SETTING OF COMPONENTS
    x0=np.arange(xInt, DW, 2*xInt)# x-coordinates of the centers of components
    y0 = np.arange(yInt,DH,2*yInt)               # coordinates of initial components' center
    xn = len(x0)
    yn = len(y0)                   # num. of component along x                
    x0=np.kron(x0,np.ones((1,2*yn)))
    y0=np.matlib.repmat(np.kron(y0,np.ones((1,2))),1,xn)  # full coordinates vector  
    N=x0.shape[1]# total number of components in the design domain
    L=np.matlib.repmat(vInt[0],1,N)# vector of the lf length of each component
    t1=np.matlib.repmat(vInt[1],1,N) # vector of the half width of component at point A
    t2=np.matlib.repmat(vInt[2],1,N) # vector of the half width of component at point B
    theta = np.matlib.repmat([vInt[3],-vInt[3]],1,int(N/2))  # vector of inclined angle 
    dd = np.vstack((x0,y0,L,t1,t2,theta))
    nDsvb = len(dd.flatten())             # number of all design variables
    nEhcp = int(nDsvb/N)                      # number design variables each component
    actComp = np.arange(0,N)               # initial set of active components 
    actDsvb = np.arange(0,nDsvb)          #initial set of active design variables
    nNd = 0                     
    PhiNd = np.array([])                           # number of non-design patch and its TDF matrix
    allPhi = np.zeros((nNod,N))   
    
    
    # SEC 5): SETTING OF MMA
    m = 1
    c = 1000*np.ones((m,1))
    d = np.zeros((m,1)) 
    a0 = 1
    a = np.zeros((m,1))
    xval = dd.flatten(order='F')
    xval = xval.reshape((xval.shape[0],1))
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin=np.vstack((0.0, 0.0, 0.1, 0.02, 0.02, -np.pi))
    xmax=np.vstack((DW, DH, 1.0, 0.1*min(DW,DH),0.1*min(DW,DH), np.pi))
    xmin=np.matlib.repmat(xmin,N,1)
    xmax=np.matlib.repmat(xmax,N,1)
    low = xmin
    upp = xmax
    nn=6*N

    def comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld):
    
        allPhiDrv=lil_matrix((nNod,nDsvb))
        for i in actComp:                      # calculating TDF of the active MMCs                                                    
            allPhi,allPhiDrv,xval,actComp,actDsvb = calc_Phi(allPhi,allPhiDrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb)
        allPhiAct = np.array(allPhi[:,actComp])                          # TDF matrix of active components
        temp = np.exp(lmd*allPhiAct)
        Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        # global TDF using K-S aggregation
        allPhiDrvAct = allPhiDrv[:,actDsvb]

        Phimaxdphi = np.kron(np.divide(temp[:,0:len(actComp)],(np.sum(temp,1)+np.spacing(1)).reshape((len(temp),1),order='F')),np.ones((1,nEhcp)))

        PhimaxDrvAct = allPhiDrvAct.multiply(Phimaxdphi)                # nodal sensitivity of global TDF
        
        #%--------------------------LP 3): Finite element analysis
        H = Heaviside(Phimax,alpha,epsilon)                            # nodal density vector 
        den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)
        U = np.zeros((nDof,1))
        nAct = len(actComp) + nNd                                # number of active components (for load path)
        #strct,loadPth = srch_ldpth(nAct,allPhiAct,Phimax,epsilon,eleNodesID,loadEle,fixEle)  # load path identification
        strct=0
        if strct == 1 :                                                  # load path existed, FEA with DOFs removal 
            if len(loadPth) == nAct:         # no islands
                denSld = den 
            else:                            # isolated components existed
                PhimaxSld = np.maximum(-1e3,np.log(np.sum(np.exp(lmd*allPhiAct[:,np.int32(loadPth)]),1))/lmd) # global TDF of components in load path

            eleLft = np.setdiff1d(np.arange(len(denSld)), np.where(denSld < alpha + np.spacing(1))[0])    # retained elements for FEA
            edofMatLft = edofMat[eleLft,:]
            freedofLft = np.setdiff1d(edofMatLft,fixDof)                    #retained DOFs for FEA
            iK1,jK1 = np.array(edofMatLft[:,sI.astype('int')]).T,np.array(edofMatLft[:,sII.astype('int')]).T
            Iar = np.sort(np.vstack((iK1.flatten(order='F'), jK1.flatten(order='F'))).T, axis=1)[:, ::-1]     # new reduced assembly indexing
            sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(denSld[eleLft],(denSld[eleLft].shape[0],1)).T)
            sK = sK.flatten(order='F')

            K = csc_matrix((sK, (Iar[:,0], Iar[:,1])), shape=(nDof, nDof))
            K = K + K.T - diags((K.diagonal()))
            K = K + csc_matrix((np.spacing(1)*np.ones(nDof), (np.arange(nDof), np.arange(nDof))), shape=(nDof, nDof))   # regularization of disconnected component
            U[freedofLft] =spsolve(K[freedofLft,:][:,freedofLft], F[freedofLft]).reshape((len(freedofLft),1))

        else:
                                                                    # no load path, regular FEA
            #print('WARNING!!! NO loading path was found!!!')
            sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(den,(den.shape[0],1)).T)
            sK = sK.flatten(order='F')
            K = csc_matrix((sK.flatten(order='F'), (Iar0[:,0], Iar0[:,1])), shape=(nDof, nDof))
            K =  K + K.T - diags((K.diagonal()))
            U[freeDof] =spsolve(K[freeDof,:][:,freeDof], F[freeDof]).reshape((len(freeDof),1))

        f0val = F.T*U/scl
        fval = sum(den)*EL*EW/(DW*DH) - volfrac
        
        return f0val,fval,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld
    
    def comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld):
        allPhiDrv=lil_matrix((nNod,nDsvb))
        for i in actComp:                      # calculating TDF of the active MMCs                                                            
            allPhi,allPhiDrv,xval,actComp,actDsvb = calc_Phi(allPhi,allPhiDrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb)
        allPhiAct = np.array(allPhi[:,actComp])                          # TDF matrix of active components
        temp = np.exp(lmd*allPhiAct)
        Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        # global TDF using K-S aggregation
        allPhiDrvAct = allPhiDrv[:,actDsvb]

        Phimaxdphi = np.kron(np.divide(temp[:,0:len(actComp)],(np.sum(temp,1)+np.spacing(1)).reshape((len(temp),1),order='F')),np.ones((1,nEhcp)))

        PhimaxDrvAct = allPhiDrvAct.multiply(Phimaxdphi)                # nodal sensitivity of global TDF


        #%--------------------------LP 3): Finite element analysis
        H = Heaviside(Phimax,alpha,epsilon)                            # nodal density vector 
        den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)
        U = np.zeros((nDof,1))
        nAct = len(actComp) + nNd                                # number of active components (for load path)
        #strct,loadPth = srch_ldpth(nAct,allPhiAct,Phimax,epsilon,eleNodesID,loadEle,fixEle)  # load path identification
        strct=0
        if strct == 1 :                                                  # load path existed, FEA with DOFs removal 
            if len(loadPth) == nAct:         # no islands
                denSld = den 
            else:                            # isolated components existed
                PhimaxSld = np.maximum(-1e3,np.log(np.sum(np.exp(lmd*allPhiAct[:,np.int32(loadPth)]),1))/lmd) # global TDF of components in load path

            eleLft = np.setdiff1d(np.arange(len(denSld)), np.where(denSld < alpha + np.spacing(1))[0])    # retained elements for FEA
            edofMatLft = edofMat[eleLft,:]
            freedofLft = np.setdiff1d(edofMatLft,fixDof)                    #retained DOFs for FEA
            iK1,jK1 = np.array(edofMatLft[:,sI.astype('int')]).T,np.array(edofMatLft[:,sII.astype('int')]).T
            Iar = np.sort(np.vstack((iK1.flatten(order='F'), jK1.flatten(order='F'))).T, axis=1)[:, ::-1]     # new reduced assembly indexing
            sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(denSld[eleLft],(denSld[eleLft].shape[0],1)).T)
            sK = sK.flatten(order='F')

            K = csc_matrix((sK, (Iar[:,0], Iar[:,1])), shape=(nDof, nDof))
            K = K + K.T - diags((K.diagonal()))
            K = K + csc_matrix((np.spacing(1)*np.ones(nDof), (np.arange(nDof), np.arange(nDof))), shape=(nDof, nDof))   # regularization of disconnected component
            U[freedofLft] =spsolve(K[freedofLft,:][:,freedofLft], F[freedofLft]).reshape((len(freedofLft),1))

        else:
                                                                    # no load path, regular FEA
            #print('WARNING!!! NO loading path was found!!!')
            sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(den,(den.shape[0],1)).T)
            sK = sK.flatten(order='F')
            K = csc_matrix((sK.flatten(order='F'), (Iar0[:,0], Iar0[:,1])), shape=(nDof, nDof))
            K =  K + K.T - diags((K.diagonal()))
            U[freeDof] =spsolve(K[freeDof,:][:,freeDof], F[freeDof]).reshape((len(freeDof),1))

        f0val = F.T*U/scl
        fval = sum(den)*EL*EW/(DW*DH) - volfrac

        #--------------------------LP 4): Sensitivity analysis
        df0dx = np.zeros((1,nDsvb))
        dfdx = np.zeros((1,nDsvb))
        delta_H = 3*(1-alpha)/(4*epsilon)*(1-Phimax**2/(epsilon**2))
        delta_H[abs(Phimax)>epsilon] = 0                              # derivative of nodal density to nodal TDF
        energy = np.sum(np.multiply(np.dot(U[edofMat].squeeze(),KE),U[edofMat].squeeze()),axis=1)
        sEner = energy.reshape((energy.shape[0],1))*np.ones((1,4))/4    
        engyNod = csc_matrix((sEner.flatten(order='F'), (eleNodesID.flatten(order='F').astype('int'), np.zeros(eleNodesID.size, dtype=int)))) #nodal form of Ue'*K0*Ue
        df0dx[:,actDsvb] = -(engyNod.multiply(delta_H.reshape((delta_H.shape[0],1))).T*csc_matrix(PhimaxDrvAct)).todense()      # sensitivity of objective function     
        dfdx[:,actDsvb] = (volNod.multiply(delta_H.reshape((delta_H.shape[0],1))).T*csc_matrix(PhimaxDrvAct)).todense()*EL*EW/(DW*DH) # sensitivity of volume constraint

        dgt = dgt0 - np.floor(np.log10(np.array([np.max(np.abs(df0dx)), np.max(np.abs(dfdx))])))    # significant digits for sens. truncation 
        df0dx = np.round(df0dx*10**dgt[0])/10**dgt[0]/scl                  # truncated scaled objective sensitivity
        dfdx  = np.round(dfdx*10**dgt[1])/10**dgt[1]                       # truncated constraint sensitivity
        dfdx=dfdx/np.max(abs(dfdx))
        df0dx=df0dx/np.max(abs(df0dx))
        
        return f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld
        
    # SEC 6): OPTIMIZATION LOOP
    loop=1
    outit = 0
    totalinner_it=0
    maxinnerinit=1
    OBJ=[]
    CONS=[]
    outeriter=0
    eeem = np.ones((m,1))
    raa0eps = 0.000001
    raaeps = 0.000001*eeem
    raa = 0.01*eeem
    raa0 = 0.01
    epsimin = 1e-09
    denSld=[0]
    change=1000

    f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi,actComp,actDsvb,allPhiDrv,denSld = comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld)
    f0val_1=f0val.copy()
    f0val_2=f0val.copy()
    criteria=((f0val_2-f0val_1)/((abs(f0val_2)+abs(f0val_1))/2))*((f0val_1-f0val)/(abs(f0val_1)+abs(f0val))/2)
    optimizer='MMA'

    while objVr5>1e-4 and loop<=maxiter:
        outeriter += 1
        criteria=((f0val_2-f0val_1)/((abs(f0val_2)+abs(f0val_1))/2))*((f0val_1-f0val)/(abs(f0val_1)+abs(f0val))/2)
        if criteria>-0.00002 and criteria<0:
            optimizer='GCMMA'  

        if optimizer=='MMA':
            xmma,_,_,_,_,_,_,_,_,low,upp = mmasub(m,nDsvb,loop,xval.reshape((xval.shape[0],1),order='F'),xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move=0.5)

        if optimizer=='GCMMA':
            # The parameters low, upp, raa0 and raa are calculated:
            low,upp,raa0,raa = asymp(outeriter,nn,xval.reshape((xval.shape[0],1),order='F'),xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx.T,dfdx)  
            # The MMA subproblem is solved at the point xval:
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,nn,iter,epsimin,xval.reshape((xval.shape[0],1),order='F'),xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx.T,a0,a,c,d)
            # The user should now calculate function values (no gradients) of the objective- and constraint
            # functions at the point xmma ( = the optimal solution of the subproblem).
            f0valnew,fvalnew,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld = comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld)

            # It is checked if the approximations are conservative:
            conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)

            # While the approximations are non-conservative (conserv=0), repeated inner iterations are made:
            innerit = 0
            if conserv == 0:
                while conserv == 0 and innerit <= maxinnerinit:
                    innerit += 1
                    totalinner_it+=1
                    # New values on the parameters raa0 and raa are calculated:
                    raa0,raa = raaupdate(xmma,xval,xmin,xmax,low,upp,f0valnew,fvalnew,f0app,fapp,raa0,raa,raa0eps,raaeps,epsimin)
                    # The GCMMA subproblem is solved with these new raa0 and raa:
                    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,nn,iter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx.T,a0,a,c,d)
                    # The user should now calculate function values (no gradients) of the objective- and 
                    # constraint functions at the point xmma ( = the optimal solution of the subproblem).
                    f0valnew,fvalnew,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv,denSld = comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld)
                    # It is checked if the approximations have become conservative:
                    conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)
                    
        change=max(abs(xval-xmma))
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()              # design variable's update
        f0val_2=f0val_1.copy()
        f0val_1=f0val.copy()
        
        f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi,actComp,actDsvb,allPhiDrv,denSld = comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,loadEle,fixEle,xval,denSld)
        OBJ.append(f0val*scl)       # scaled objective function 
        CONS.append(fval + volfrac)             # volume constraint    

        # ---  Plotting current design
        #Plot components
        if plotting=="contour":
            fig = plt.figure()
            ax = plt.subplot(111)
            colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']
            for i, color in zip(range(0,N), colors):
                ax.contourf(x,y,np.flipud(allPhi[:,i].reshape((nely+1,nelx+1),order='F')),[0,1],colors=color)
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

            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show() 

        if plotting== "component":
            for i in range(0,N):
                plt.contour(x,y,allPhi[:,i].reshape(nely+1,nelx+1,order='F'),[0])
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
        
        
        if loop>=15 and (fval/volfrac)<5e-4:
            objVr5 = abs(max(abs(OBJ[-15:] - np.mean(OBJ[-15:]))) / np.mean(OBJ[-15:]))
            
        print('Optim: ',optimizer,'It.: ',loop+totalinner_it, ' Obj.: ',f0val, ' Vol.: ', fval, 'ch.:', objVr5, 'xval_change', change)    
        print('Oscillation criteria: ',criteria)
        print(fval/volfrac)
        print("Volume fraction: ",fval," Desired: ",volfrac)
        loop+=1

