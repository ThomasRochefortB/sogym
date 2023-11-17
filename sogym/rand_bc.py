import numpy as np
import random

def calculate_support_orientation(selected_boundary):
    if selected_boundary == 'left' or selected_boundary == 'right':
        return 0  # Primary orientation for left and right is 0 degrees
    elif selected_boundary == 'bottom':
        return 270  # Primary orientation for bottom is 270 degrees
    elif selected_boundary == 'top':
        return 90  # Primary orientation for top is 90 degrees

def filter_load_orientations(support_orientation):
    valid_orientations = [i for i in range(0, 360, 15)]
    exclude_range = [(support_orientation + angle) % 360 for angle in range(-45, 46)]
    exclude_range += [(support_orientation + 180 + angle) % 360 for angle in range(-45, 46)]
    return [o for o in valid_orientations if o not in exclude_range]
def gen_randombc(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    # dx and dy will be sampled from a uniform distribution between 1.0 and 2.0. The number will only have one digit (ex: 1.1,1.2, ...)
    dx = np.round(np.random.uniform(1.0,2.0),1)
    dy = np.round(np.random.uniform(1.0,2.0),1)

    # The resolution of the mesh will implement a fixed element size of 0.01 (100 elements per 1.0 unit)
    nelx = int(50 * dx)
    nely = int(50 * dy)

    # Sample the desired volume fraction between 0.2 and 0.4
    volume_fraction = np.round(np.random.uniform(0.2,0.4),2)

    # Step 1: Select external boundary that will be supported and the type of support chosen:
    boundaries = ['left', 'right', 'bottom', 'top']
    selected_boundary = np.random.choice(boundaries)
    support_type = ['fully'] #, 'simple']
    selected_type = np.random.choice(support_type)
    
    # Step 2: Select fully-supported boundary length and position
    boundary_length = np.random.uniform(0.25, 0.75) * nely if selected_boundary in ['left', 'right'] else np.random.uniform(0.25, 0.75) * nelx
    boundary_position = np.random.uniform(0, 1-(boundary_length/nely)) if selected_boundary in ['left', 'right'] else np.random.uniform(0, 1-(boundary_length/nelx))
    boundary_length = np.round(boundary_length)
    boundary_position = np.round(boundary_position,2)


    # Select a random number of loads between 1 and 2:
    # n_loads = np.random.randint(1,4)
    n_loads = 1
    # Step 3: Select the degrees of freedom that are affected by the boundary condition:
    nodes_matrix = np.zeros((nely+1, nelx+1))
    if selected_boundary == 'left':
        nodes_matrix[int(boundary_position*nely):int(boundary_position*nely)+int(boundary_length), 0] = 1
    elif selected_boundary == 'right':
        nodes_matrix[int(boundary_position*nely):int(boundary_position*nely)+int(boundary_length), nelx] = 1
    elif selected_boundary == 'bottom':
        nodes_matrix[nely, int(boundary_position*nelx):int(boundary_position*nelx)+int(boundary_length)] = 1
    elif selected_boundary == 'top':
        nodes_matrix[0, int(boundary_position*nelx):int(boundary_position*nelx)+int(boundary_length)] = 1

    #generate the fixeddofs:
    fixednode=np.argwhere(nodes_matrix.reshape((((nely+1))*((nelx+1))),order='F')==1)
    fixeddofs=[]
    if selected_boundary == 'left' or selected_boundary == 'right':
        if selected_type == 'simple':
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
        else:
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
                fixeddofs.append((2*fixednode[i])+1)

    elif selected_boundary == 'bottom' or selected_boundary == 'top':
        if selected_type == 'simple':
            for i in range(0,len(fixednode)):
                fixeddofs.append((2*fixednode[i])+1)
        else:
            for i in range(0,len(fixednode)):
                fixeddofs.append(2*fixednode[i])
                fixeddofs.append((2*fixednode[i])+1)


    # Generate n_loads random position for each load and ensure they are different:
    load_position = np.round(np.random.uniform(0, 0.0,size=n_loads),2)
    if n_loads==1:
        while len(np.unique(load_position)) != n_loads:
            load_position = np.round(np.random.uniform(0, 0.99,size=n_loads),2)
    else:
        while len(np.unique(load_position)) != n_loads or np.min(np.abs(np.subtract.outer(load_position,load_position))[np.triu_indices(n_loads,1)]) < 0.1:
            load_position = np.round(np.random.uniform(0, 0.99,size=n_loads),2)

        
    if selected_type == 'fully':
        # Select a random orientation for the load(s):
        # In your gen_randombc function:
        support_orientation = calculate_support_orientation(selected_boundary)
        valid_load_orientations = filter_load_orientations(support_orientation)
        load_orientation = np.random.choice(valid_load_orientations, size=n_loads)
        magnitude_x = np.cos(np.radians(load_orientation))
        magnitude_y = np.sin(np.radians(load_orientation))

    elif selected_type == 'simple':
        if selected_boundary == 'left':
            load_orientation = [180]*n_loads
        elif selected_boundary == 'right':
            load_orientation = [0]*n_loads
        elif selected_boundary == 'bottom':
            load_orientation = [270]*n_loads
        elif selected_boundary == 'top':
            load_orientation = [90]*n_loads

        magnitude_x = np.cos(np.radians(load_orientation))
        magnitude_y = np.sin(np.radians(load_orientation))

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
    for node in loadnode:
        loaddof_x.append(2*node)
        loaddof_y.append((2*node)+1)


    # I need to normalize the boundary_length value based on the length of the selected boundary:
    if selected_boundary == 'left' or selected_boundary == 'right':
        boundary_length_norm = boundary_length/nely
    elif selected_boundary == 'bottom' or selected_boundary == 'top':
        boundary_length_norm = boundary_length/nelx


    out_dict = {
                'selected_boundary':boundaries.index(selected_boundary) / len(boundaries),                # Convert selected_boundary to integers (0: left, 1: right, 2: bottom, 3: top)
                'support_type':support_type.index(selected_type),                 # Convert support_type to integers (0: fully, 1: simple)
                'boundary_length':boundary_length_norm,   # A single float value normalized by the length of the selected boundary
                'boundary_position':boundary_position, # A single float value
                'n_loads':n_loads,  # A single int value
                'load_position':load_position,  # A list of float values
                'load_orientation':np.array(load_orientation)/360, # A list of float values normalized by 360 degrees.
                'fixeddofs':fixeddofs,
                'loaddof_x':np.array(loaddof_x).flatten(),
                'loaddof_y':np.array(loaddof_y).flatten(),
                'magnitude_x':magnitude_x,
                'magnitude_y':magnitude_y,
                'volfrac':volume_fraction,
                'loadnode':loadnode,
                'fixednode':fixednode
                }
    if len(np.array(loaddof_x).flatten()) != len(magnitude_x):
        print('mismatch in rand_bc!!', loaddof_x,magnitude_x,load_position,n_loads)
    return dx, dy, nelx, nely, out_dict

def generate_prompt(conditions,dx,dy):

    dict ={}
    dict['support'] = {'type':conditions['support_type'], 'boundary':conditions['selected_boundary'], 'length':conditions['boundary_length'], 'position':conditions['boundary_position']}
    dict['domain'] = {'type':'rectangular',
                      'x_dimension': dx,
                      'y_dimension': dy}
    
    dict['constraints'] = {'desired_volume_constraint':conditions['volfrac']}

    for i in range(conditions['n_loads']):
        dict['load_'+str(i)] = {'type':'point_load',
                                'position':conditions['load_position'][i],
                                'orientation':conditions['load_orientation'][i],
                                'magnitude_x':conditions['magnitude_x'][i],
                                'magnitude_y':conditions['magnitude_y'][i]}
    return str(dict)
