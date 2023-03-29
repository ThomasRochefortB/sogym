import numpy as np

def gen_randombc():

    # dx and dy will be sampled from a uniform distribution between 1.0 and 2.0. The number will only have one digit (ex: 1.1,1.2, ...)
    dx = np.round(np.random.uniform(1.0,2.0),1)
    dy = np.round(np.random.uniform(1.0,2.0),1)

    # The resolution of the mesh will implement a fixed element size of 0.01 (100 elements per 1.0 unit)
    nelx = int(100 * dx)
    nely = int(100 * dy)

    # Sample the desired volume fraction between 0.2 and 0.4
    volume_fraction = np.round(np.random.uniform(0.2,0.4),2)

    # Step 1: Select external boundary
    boundaries = ['left', 'right', 'bottom', 'top']
    selected_boundary = np.random.choice(boundaries)
    support_type = np.random.choice(['fully', 'simple'])

    # Step 2: Select fully-supported boundary length and position
    boundary_length = np.random.uniform(0.25, 0.5) * nely if selected_boundary in ['left', 'right'] else np.random.uniform(0.25, 0.5) * nelx
    boundary_position = np.random.uniform(0, 1-boundary_length/nely) if selected_boundary in ['left', 'right'] else np.random.uniform(0, 1-boundary_length/nelx)
    boundary_length = np.round(boundary_length)
    boundary_position = np.round(boundary_position,2)


    # Select a random number of loads between 1 and 3:
    n_loads = np.random.randint(1,4)
    # Step 3: Select the degrees of freedom that are affected by the boundary condition:
    nodes_matrix = np.zeros((nely+1, nelx+1))
    if selected_boundary == 'left':
        nodes_matrix[int(boundary_position*nely):int(boundary_position*nely)+int(boundary_length*nely), 0] = 1
    elif selected_boundary == 'right':
        nodes_matrix[int(boundary_position*nely):int(boundary_position*nely)+int(boundary_length*nely), nelx] = 1
    elif selected_boundary == 'bottom':
        nodes_matrix[nely, int(boundary_position*nelx):int(boundary_position*nelx)+int(boundary_length*nelx)] = 1
    elif selected_boundary == 'top':
        nodes_matrix[0, int(boundary_position*nelx):int(boundary_position*nelx)+int(boundary_length*nelx)] = 1

    #generate the fixeddofs:
    fixednode=np.argwhere(nodes_matrix.reshape((((nely+1))*((nelx+1))),order='F')==1)
    fixeddofs=[]
    if selected_boundary == 'left' or selected_boundary == 'right':
        if support_type == 'simple':
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
        else:
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
                fixeddofs.append((2*fixednode[i])+1)

    elif selected_boundary == 'bottom' or selected_boundary == 'top':
        if support_type == 'simple':
            for i in range(0,len(fixednode)):
                fixeddofs.append((2*fixednode[i])+1)
        else:
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
                fixeddofs.append((2*fixednode[i])+1)


    # Generate a random position for each load:
    load_position = np.round(np.random.uniform(0, 1,size=n_loads),2)
    if support_type == 'fully':
        # Select a random orientation for the load:
        load_orientation = np.random.choice([0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345],size=n_loads)
        magnitude_x = np.array([np.cos(np.radians(load_orientation))])
        magnitude_y = np.array([np.sin(np.radians(load_orientation))])

    elif support_type == 'simple':
        if selected_boundary == 'left':
            load_orientation = [180]*n_loads
        elif selected_boundary == 'right':
            load_orientation = [0]*n_loads
        elif selected_boundary == 'bottom':
            load_orientation = [270]*n_loads
        elif selected_boundary == 'top':
            load_orientation = [90]*n_loads

        magnitude_x = np.array([np.cos(np.radians(load_orientation))])
        magnitude_y = np.array([np.sin(np.radians(load_orientation))])

    # Select a random position for the load:
    for i in range(n_loads):
        if selected_boundary == 'left':
            nodes_matrix[int(load_position[i]*nely), nelx] = 2
        elif selected_boundary == 'right': 
            nodes_matrix[int(load_position[i]*nely), 0] = 2
        elif selected_boundary == 'bottom':
            nodes_matrix[0, int(load_position[i]*nelx)] = 2
        elif selected_boundary == 'top':
            nodes_matrix[nely, int(load_position[i]*nelx)] = 2

    #generate the loaddofs:
    loadnode=np.argwhere(nodes_matrix.reshape((((nely+1))*((nelx+1))),order='F')==2)
    loaddof_x=[]
    loaddof_y=[]
    for i in range(0,len(loadnode)):
        loaddof_x.append(2*loadnode)
        loaddof_y.append((2*loadnode)+1)

    out_dict = {
                'selected_boundary':selected_boundary,
                'support_type':support_type,
                'boundary_length':boundary_length,
                'boundary_position':boundary_position,
                'n_loads':n_loads,
                'load_position':load_position,
                'load_orientation':load_orientation,
                'fixeddofs':fixeddofs,
                'loaddof_x':loaddof_x,
                'loaddof_y':loaddof_y,
                'magnitude_x':magnitude_x,
                'magnitude_y':magnitude_y,
                'volfrac':volume_fraction,
                'loadnode':loadnode,
                'fixednode':fixednode
                }
    return dx, dy, nelx, nely, out_dict

