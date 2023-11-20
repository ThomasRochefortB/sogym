# A 200 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# Updated by Niels Aage February 2016
import os
# Disable multithreading for various libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # For OpenBLAS

import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt 
import cvxopt.cholmod


def run_simp(nelx,nely,BC_dict,penal,rmin,ft, cfg=None, verbose=1):
	
	if cfg is None:
		cfg = {
			'optimizer': 'OC',
			'Emin': 1e-9,
			'Emax': 1.0,
			'maxiter': 1000,
			'convergence_threshold': 1e-2
		}

	volfrac = BC_dict['volfrac']
	loaddof_x = BC_dict['loaddof_x']
	loaddof_y = BC_dict['loaddof_y']
	magnitude_x = BC_dict['magnitude_x']
	magnitude_y = BC_dict['magnitude_y']
	if verbose==1:
		print("Minimum compliance problem with OC")
		print("ndes: " + str(nelx) + " x " + str(nely))
		print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
		print("Filter method: " + ["Sensitivity based","Density based"][ft])

	# Max and min stiffness
	Emin=cfg['Emin']
	Emax=cfg['Emax']

	# dofs:
	ndof = 2*(nelx+1)*(nely+1)

	# Allocate design variables (as array), initialize and allocate sens.
	x=volfrac * np.ones(nely*nelx,dtype=float)
	xold=x.copy()
	xPhys=x.copy()

	g=0 # must be initialized to use the NGuyen/Paulino OC approach
	dc=np.zeros((nely,nelx), dtype=float)

	# FE: Build the index vectors for the for coo matrix format.
	KE=lk()
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
	# Construct the index pointers for the coo format
	iK = np.kron(edofMat,np.ones((8,1))).flatten()
	jK = np.kron(edofMat,np.ones((1,8))).flatten()    

	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
	cc=0
	for i in range(nelx):
		for j in range(nely):
			row=i*nely+j
			kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
			kk2=int(np.minimum(i+np.ceil(rmin),nelx))
			ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
			ll2=int(np.minimum(j+np.ceil(rmin),nely))
			for k in range(kk1,kk2):
				for l in range(ll1,ll2):
					col=k*nely+l
					fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
					iH[cc]=row
					jH[cc]=col
					sH[cc]=np.maximum(0.0,fac)
					cc=cc+1
	# Finalize assembly and convert to csc format
	H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
	Hs=H.sum(1)

	# BC's and support
	dofs=np.arange(2*(nelx+1)*(nely+1))
	fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
	fixed=BC_dict['fixeddofs']      # fixed nodes

	free=np.setdiff1d(dofs,fixed)

	# Solution and RHS vectors
	f=np.zeros((ndof,1))
	u=np.zeros((ndof,1))

	# Set load
	# Set load

	# Create load vectors
	f_x = np.zeros((ndof, 1))
	f_y = np.zeros((ndof, 1))


	# Assign the load magnitudes to the corresponding degrees of freedom
	f_x[loaddof_x, 0] = magnitude_x
	f_y[loaddof_y, 0] = magnitude_y

	# Combine the load vectors
	f = f_x + f_y
	
	#plt.ion() # Ensure that redrawing is possible
	# fig,ax = plt.subplots()
	# im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
	# interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0),origin='lower')
    


	loop=0
	change=1
	dv = np.ones(nely*nelx)
	dc = np.ones(nely*nelx)
	ce = np.ones(nely*nelx)
	while change>cfg['convergence_threshold'] and loop<cfg['maxiter']:
		loop=loop+1
		# Setup and solve FE problem
		sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
		# Remove constrained dofs from matrix and convert to coo
		K = deleterowcol(K,fixed,fixed).tocoo()
		# Solve system 
		K = cvxopt.spmatrix(K.data,K.row.astype(int),K.col.astype(int))
		B = cvxopt.matrix(f[free,0])
		cvxopt.cholmod.linsolve(K,B)
		u[free,0]=np.array(B)[:,0] 

		# Objective and sensitivity
		ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
		obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
		dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce

		dv[:] = np.ones(nely*nelx)
		# Sensitivity filtering:
		if ft==0:
			dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
		elif ft==1:
			dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
			dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]

		# Optimality criteria
		xold[:]=x
		(x[:],g)=oc(nelx,nely,x,volfrac,dc,dv,g)

		# Filter design variables
		if ft==0:   xPhys[:]=x
		elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
	    
		# Compute the change by the inf. norm
		change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)

		# Write iteration history to screen (req. Python 2.6 or newer)
		if verbose==1:
			print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
						loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))


	# Make sure the plot stays and that the shell remains	
	return xPhys.reshape((nelx,nely)).T, obj, loop, cfg
    
#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)

def oc(nelx,nely,x,volfrac,dc,dv,g):
	l1=0
	l2=1e9
	move=0.2
	# reshape to perform vector operations
	xnew=np.zeros(nelx*nely)

	while (l2-l1)/(l1+l2)>1e-3:
		lmid=0.5*(l2+l1)
		xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
		gt=g+np.sum((dv*(xnew-x)))
		if gt>0 :
			l1=lmid
		else:
			l2=lmid
	return (xnew,gt)
    
def deleterowcol(A, delrow, delcol):
	# Assumes that matrix is in symmetric csc form !
	m = A.shape[0]
	keep = np.delete (np.arange(0, m), delrow)
	A = A[keep, :]
	keep = np.delete (np.arange(0, m), delcol)
	A = A[:, keep]
	return A    