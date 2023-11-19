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

# Topology description function and derivatives
def calc_Phi(allPhi,allPhidrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb,minSz,epsilon):
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
    if np.min(dd[3:5])/minSz > 0.1 and min(abs(allPhi[:,i])) < epsilon: #and dd[2] > minSz:
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
    else:                 # deleting tiny componennp.spacing(1)t and removing it from active sets
        print(['The {}-th component is too small! DELETE it!!!'.format(i)])   
        allPhi[:,i] = -1e3
        xval[(i)*nEhcp+3] = 0
        xval[(i)*nEhcp+4] = 0
        actComp = np.setdiff1d(actComp,i)
        actDsvb = np.setdiff1d(actDsvb, np.arange(nEhcp*(i+1)-nEhcp, nEhcp*(i+1)))
    return [allPhi,allPhidrv,xval,actComp,actDsvb]


def run_mmc(BC_dict,nelx,nely,dx,dy,plotting='component',verbose=0):   ## Probably need to add xmin and xmax
    xInt = 0.165*dx  #0.125 for 8 x 8, 0.165 for 3x3, 0.25 for 2x2
    yInt = 0.165*dy
    vInt = [0.3*min(dx,dy), 0.03, 0.03, np.arcsin(0.7)]
    E = 1.0 #Young's modulus
    nu = 0.3 #Poisson ratio
    h = 1 #thickness                              
    dgt0 = 5 #significant digit of sens.
    scl = 1 #scale factor for obj                                           
    p = 6   #power of super ellipsoid
    lmd = 100    #power of KS aggregation   (default 100)                                   
    maxiter = 500 # maximum number of iterations                                       
    objVr5 = 1.0  # initial relative variat. of obj. last 5 iterations

    ## Setting of FE discretization
    nEle = nelx*nely              # number of finite elements
    nNod = (nelx+1)*(nely+1)      # number of nodes
    nDof = 2*(nelx+1)*(nely+1)    # number of degree of freedoms
    EL = dx/nelx                  # length of finite elements
    EW = dy/nely                  # width of finite elements
    minSz = min([EL,EW])*3          # minimum size of finite elements
    alpha = 1e-9                  # void density
    epsilon = 0.2            # regularization term in Heaviside (default 0.2)
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
    x,y=np.meshgrid(np.linspace(0, dx,nelx+1),np.linspace(0,dy,nely+1))                # coordinates of nodal points
    LSgrid={"x":x.flatten(order='F'),"y":y.flatten(order='F')}
    volNod=csc_matrix((np.ones(eleNodesID.size)/4,(eleNodesID.flatten(order='F'),np.zeros(eleNodesID.size))),shape=(nNod,1))                       # weight of each node in volume calculation 

#  3): LOADS, DISPLACEMENT BOUNDARY CONDITIONS (2D cantilever beam example)
    volfrac=BC_dict['volfrac']
    magnitude_x=BC_dict['magnitude_x']
    magnitude_y=BC_dict['magnitude_y']
    loaddof_x=BC_dict['loaddof_x']                 # loaded dofs
    loaddof_y=BC_dict['loaddof_y']                 # loaded dofs
    fixDof=BC_dict['fixeddofs']      # fixed nodes
    
    freeDof = np.setdiff1d(np.arange(nDof),fixDof)         # index of free dofs
    F_x=csc_matrix((magnitude_x, (loaddof_x, np.zeros_like(loaddof_x))), shape=(nDof, 1))
    F_y=csc_matrix((magnitude_y, (loaddof_y, np.zeros_like(loaddof_y))), shape=(nDof, 1))
    F=F_x+F_y
    
    
    #  4): INITIAL SETTING OF COMPONENTS
    x0 = np.arange(xInt, dx, 2*xInt)# x-coordinates of the centers of components
    y0 = np.arange(yInt, dy, 2*yInt)               # coordinates of initial components' center
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
    xmin=np.vstack((0.0, 0.0, 0.0, 0.00, 0.00, -np.pi))
    xmax=np.vstack((dx, dy, 0.7*min(dx,dy), 0.05*min(dx,dy),0.05*min(dx,dy), np.pi))
    xmin=np.matlib.repmat(xmin,N,1)
    xmax=np.matlib.repmat(xmax,N,1)
    low = xmin
    upp = xmax
    nn=6*N

    def comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld):
    
        allPhiDrv=lil_matrix((nNod,nDsvb))
        for i in actComp:                      # calculating TDF of the active MMCs                                                    
            allPhi,allPhiDrv,xval,actComp,actDsvb = calc_Phi(allPhi,allPhiDrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb,minSz,epsilon)
        allPhiAct = np.array(allPhi[:,actComp])                          # TDF matrix of active components
        temp = np.exp(lmd*allPhiAct)
        # temp = np.where(temp==0,1e-08,temp)

        Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        # global TDF using K-S aggregation
        allPhiDrvAct = allPhiDrv[:,actDsvb]

        Phimaxdphi = np.kron(np.divide(temp[:,0:len(actComp)],(np.sum(temp,1)+np.spacing(1)).reshape((len(temp),1),order='F')),np.ones((1,nEhcp)))

        PhimaxDrvAct = allPhiDrvAct.multiply(Phimaxdphi)                # nodal sensitivity of global TDF
        
        #%--------------------------LP 3): Finite element analysis
        H = Heaviside(Phimax,alpha,epsilon)                            # nodal density vector 
        den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)
        U = np.zeros((nDof,1))
        nAct = len(actComp) + nNd                                # number of active components (for load path)
        sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(den,(den.shape[0],1)).T)
        sK = sK.flatten(order='F')
        K = csc_matrix((sK.flatten(order='F'), (Iar0[:,0], Iar0[:,1])), shape=(nDof, nDof))
        K =  K + K.T - diags((K.diagonal()))
        U[freeDof] =spsolve(K[freeDof,:][:,freeDof], F[freeDof]).reshape((len(freeDof),1))

        f0val = F.T*U/scl
        fval = sum(den)*EL*EW/(dx*dy) - volfrac
        
        return f0val,fval,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld, den
    
    def comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld):
        allPhiDrv=lil_matrix((nNod,nDsvb))
        for i in actComp:                      # calculating TDF of the active MMCs                                                            
            allPhi,allPhiDrv,xval,actComp,actDsvb = calc_Phi(allPhi,allPhiDrv,xval,i,LSgrid,p,nEhcp,actComp,actDsvb,minSz,epsilon)
        allPhiAct = np.array(allPhi[:,actComp])                          # TDF matrix of active components
        temp = np.exp(lmd*allPhiAct)
        # temp = np.where(temp==0,1e-08,temp)

        Phimax = np.maximum(-1e3,np.log(np.sum(temp,1))/lmd)                        # global TDF using K-S aggregation
        allPhiDrvAct = allPhiDrv[:,actDsvb]

        Phimaxdphi = np.kron(np.divide(temp[:,0:len(actComp)],(np.sum(temp,1)+np.spacing(1)).reshape((len(temp),1),order='F')),np.ones((1,nEhcp)))

        PhimaxDrvAct = allPhiDrvAct.multiply(Phimaxdphi)                # nodal sensitivity of global TDF


        #%--------------------------LP 3): Finite element analysis
        H = Heaviside(Phimax,alpha,epsilon)                            # nodal density vector 
        den = np.sum(H[eleNodesID.astype('int')],1)/4                                 # elemental density vector (for volume)
        U = np.zeros((nDof,1))
        nAct = len(actComp) + nNd                                # number of active components (for load path)
                                                                            # no load path, regular FEA
        sK = np.multiply(np.reshape(Ke.flatten(order='F'),(Ke.flatten(order='F').shape[0],1)),np.reshape(den,(den.shape[0],1)).T)
        sK = sK.flatten(order='F')
        K = csc_matrix((sK.flatten(order='F'), (Iar0[:,0], Iar0[:,1])), shape=(nDof, nDof))
        K =  K + K.T - diags((K.diagonal()))
        U[freeDof] =spsolve(K[freeDof,:][:,freeDof], F[freeDof]).reshape((len(freeDof),1))

        f0val = F.T*U/scl
        fval = sum(den)*EL*EW/(dx*dy) - volfrac

        #--------------------------LP 4): Sensitivity analysis
        df0dx = np.zeros((1,nDsvb))
        dfdx = np.zeros((1,nDsvb))
        delta_H = 3*(1-alpha)/(4*epsilon)*(1-Phimax**2/(epsilon**2))
        delta_H[abs(Phimax)>epsilon] = 0                              # derivative of nodal density to nodal TDF
        energy = np.sum(np.multiply(np.dot(U[edofMat].squeeze(),KE),U[edofMat].squeeze()),axis=1)
        sEner = energy.reshape((energy.shape[0],1))*np.ones((1,4))/4    
        engyNod = csc_matrix((sEner.flatten(order='F'), (eleNodesID.flatten(order='F').astype('int'), np.zeros(eleNodesID.size, dtype=int)))) #nodal form of Ue'*K0*Ue
        df0dx[:,actDsvb] = -(engyNod.multiply(delta_H.reshape((delta_H.shape[0],1))).T*csc_matrix(PhimaxDrvAct)).todense()      # sensitivity of objective function     
        dfdx[:,actDsvb] = (volNod.multiply(delta_H.reshape((delta_H.shape[0],1))).T*csc_matrix(PhimaxDrvAct)).todense()*EL*EW/(dx*dy) # sensitivity of volume constraint

        dgt = dgt0 - np.floor(np.log10(np.array([np.max(np.abs(df0dx)), np.max(np.abs(dfdx))])))    # significant digits for sens. truncation 
        df0dx = np.round(df0dx*10**dgt[0])/10**dgt[0]/scl                  # truncated scaled objective sensitivity
        dfdx  = np.round(dfdx*10**dgt[1])/10**dgt[1]                       # truncated constraint sensitivity
        dfdx=dfdx/np.max(abs(dfdx))
        df0dx=df0dx/np.max(abs(df0dx))
        
        return f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld, den
        
    # SEC 6): OPTIMIZATION LOOP
    loop=1
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

    f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi,actComp,actDsvb,allPhiDrv,denSld, den = comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld)
    f0val_1=f0val.copy()
    f0val_2=f0val.copy()
    criteria=((f0val_2-f0val_1)/((abs(f0val_2)+abs(f0val_1))/2))*((f0val_1-f0val)/(abs(f0val_1)+abs(f0val))/2)
    optimizer='MMA'

    while objVr5>1e-4 and loop<=maxiter:
        outeriter += 1
        criteria=((f0val_2-f0val_1)/((abs(f0val_2)+abs(f0val_1))/2))*((f0val_1-f0val)/(abs(f0val_1)+abs(f0val))/2)
        if criteria>-0.000002 and criteria<0:
            optimizer='GCMMA'  

        if optimizer=='MMA':
            xmma,_,_,_,_,_,_,_,_,low,upp = mmasub(m,nDsvb,loop,xval.reshape((xval.shape[0],1),order='F'),xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move=1.0)

        if optimizer=='GCMMA':
            # The parameters low, upp, raa0 and raa are calculated:
            low,upp,raa0,raa = asymp(outeriter,nn,xval.reshape((xval.shape[0],1),order='F'),xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx.T,dfdx)  
            # The MMA subproblem is solved at the point xval:
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,nn,iter,epsimin,xval.reshape((xval.shape[0],1),order='F'),xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx.T,a0,a,c,d)
            # The user should now calculate function values (no gradients) of the objective- and constraint
            # functions at the point xmma ( = the optimal solution of the subproblem).
            f0valnew,fvalnew,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv, denSld, den = comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld)

            # It is checked if the approximations are conservative:
            print()
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
                    f0valnew,fvalnew,U,H,Phimax,allPhi, actComp, actDsvb ,allPhiDrv,denSld, den = comp(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld)
                    # It is checked if the approximations have become conservative:
                    conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)
                    
        change=max(abs(xval-xmma))
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()              # design variable's update
        f0val_2=f0val_1.copy()
        f0val_1=f0val.copy()
        
        f0val,df0dx,fval,dfdx,U,H,Phimax,allPhi,actComp,actDsvb,allPhiDrv,denSld, den = comp_deriv(nNod,nDsvb,actComp,allPhi,LSgrid,p,nEhcp,epsilon,actDsvb,minSz,lmd,alpha,eleNodesID,nNd,xval,denSld)
        OBJ.append(f0val*scl)       # scaled objective function 
        CONS.append(fval + volfrac)             # volume constraint    

        # ---  Plotting current design
        #Plot components
        if plotting=="contour":
            plt.rcParams["figure.figsize"] = (5*dx,5*dy)

            fig = plt.figure()
            ax = plt.subplot(111)
            colors = ['yellow','g','r','c','m','y','black','orange','pink','cyan','slategrey','wheat','purple','mediumturquoise','darkviolet','orangered']*10
            for i, color in zip(range(0,N), colors):
                ax.contourf(x,y,allPhi[:,i].reshape((nely+1,nelx+1),order='F'),[0,1],colors=color)
            
            # Add a rectangle to show the domain boundary:
            ax.add_patch(plt.Rectangle((0,0),dx, dy,
                                        clip_on=False,linewidth = 1,fill=False))
            
            if BC_dict['selected_boundary']==0.0:  # Left boundary
                # Add a blue rectangle to show the support 
                ax.add_patch(plt.Rectangle(xy = (0.0,dy*(BC_dict['boundary_position'])),
                                        width = BC_dict['boundary_length']*dy, 
                                        height = 0.1,
                                        angle = 90,
                                        hatch='/',
                                            clip_on=False,
                                            linewidth = 0))

                for i in range(BC_dict['n_loads']):
                    ax.arrow(dx-(BC_dict['magnitude_x'][i]*0.2),dy*(BC_dict['load_position'][i])-BC_dict['magnitude_y'][i]*0.2,
                                dx= BC_dict['magnitude_x'][i]*0.2,
                                dy = BC_dict['magnitude_y'][i]*0.2,
                                width=0.2/8,
                                length_includes_head=True,
                                head_starts_at_zero=False)
                    
            elif BC_dict['selected_boundary']==0.25: # Right boundary
                # Add a blue rectangle to show the support 
                ax.add_patch(plt.Rectangle(xy = (dx+0.1,dy*(BC_dict['boundary_position'])),
                                        width = BC_dict['boundary_length']*dy, 
                                        height = 0.1,
                                        angle = 90,
                                        hatch='/',
                                            clip_on=False,
                                            linewidth = 0))

                for i in range(BC_dict['n_loads']):
                    ax.arrow(0.0-(BC_dict['magnitude_x'][i]*0.2),dy*(BC_dict['load_position'][i])-BC_dict['magnitude_y'][i]*0.2,
                                dx= BC_dict['magnitude_x'][i]*0.2,
                                dy = BC_dict['magnitude_y'][i]*0.2,
                                width=0.2/8,
                                length_includes_head=True,
                                head_starts_at_zero=False)
            elif BC_dict['selected_boundary']==0.5: # Bottom boundary
                # Add a blue rectangle to show the support 
                ax.add_patch(plt.Rectangle(xy = (dx*BC_dict['boundary_position'],dy),
                                        width = BC_dict['boundary_length']*dx, 
                                        height = 0.1,
                                        angle = 0.0,
                                        hatch='/',
                                            clip_on=False,
                                            linewidth = 0))

                for i in range(BC_dict['n_loads']):
                    ax.arrow(dx*(BC_dict['load_position'][i])-BC_dict['magnitude_x'][i]*0.2,-(BC_dict['magnitude_y'][i]*0.2),
                                dx= BC_dict['magnitude_x'][i]*0.2,
                                dy = BC_dict['magnitude_y'][i]*0.2,
                                width=0.2/8,
                                length_includes_head=True,
                                head_starts_at_zero=False)
                    
            elif BC_dict['selected_boundary']==0.75: # Top boundary
                # Add a blue rectangle to show the support 
                ax.add_patch(plt.Rectangle(xy = (dx*BC_dict['boundary_position'],-0.1),
                                        width = BC_dict['boundary_length']*dx, 
                                        height = 0.1,
                                        angle = 0.0,
                                        hatch='/',
                                            clip_on=False,
                                            linewidth = 0))

                for i in range(BC_dict['n_loads']):
                    ax.arrow(dx*(BC_dict['load_position'][i])-BC_dict['magnitude_x'][i]*0.2,dy-(BC_dict['magnitude_y'][i]*0.2),
                                dx= BC_dict['magnitude_x'][i]*0.2,
                                dy = BC_dict['magnitude_y'][i]*0.2,
                                width=0.2/8,
                                length_includes_head=True,
                                head_starts_at_zero=False)

            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.title(str(loop)+optimizer+str(objVr5))
            plt.show() 

        if plotting== "component":
            for i in range(0,N):
                plt.contour(x,y,allPhi[:,i].reshape(nely+1,nelx+1,order='F'),[0])
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
        
        if loop>=15 and (fval/volfrac)<1e-2:
            objVr5 = abs(max(abs(OBJ[-15:] - np.mean(OBJ[-15:]))) / np.mean(OBJ[-15:]))
        
        if verbose != 0:
            print('Optim: ',optimizer,'It.: ',loop+totalinner_it, ' Obj.: ',f0val, ' Vol.: ', fval, 'ch.:', objVr5, 'xval_change', change)    
            print('Oscillation criteria: ',criteria)
            print(fval/volfrac)
            print("Volume fraction: ",fval," Desired: ",volfrac)
        loop+=1
    return xval.squeeze(), f0val.squeeze(),loop+totalinner_it, H, Phimax, allPhi, den, N

