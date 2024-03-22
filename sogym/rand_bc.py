import numpy as np
import random

def calculate_support_orientation(selected_boundary):
    orientations = {
        'left': 0,
        'right': 0,
        'bottom': 270,
        'top': 90
    }
    return orientations[selected_boundary]

def filter_load_orientations(support_orientation):
    valid_orientations = [i for i in range(0, 360, 15)]
    exclude_range = [(support_orientation + angle) % 360 for angle in range(-45, 46)]
    exclude_range += [(support_orientation + 180 + angle) % 360 for angle in range(-45, 46)]
    return [o for o in valid_orientations if o not in exclude_range]

def gen_randombc(seed, resolution=100):
    random.seed(seed)
    np.random.seed(seed)
    
    dx = np.round(np.random.uniform(1.0, 2.0), 1)
    dy = np.round(np.random.uniform(1.0, 2.0), 1)
    nelx = int(resolution * dx)
    nely = int(resolution * dy)

    volume_fraction = np.round(np.random.uniform(0.3, 0.5), 2)

    boundaries = ['left', 'right', 'bottom', 'top']
    selected_boundary = np.random.choice(boundaries)
    support_type = ['fully']
    selected_type = np.random.choice(support_type)

    boundary_length = np.random.uniform(0.5, 0.75) * (nely if selected_boundary in ['left', 'right'] else nelx)
    boundary_position = np.random.uniform(0, 1 - (boundary_length / (nely if selected_boundary in ['left', 'right'] else nelx)))
    boundary_length = np.round(boundary_length)
    boundary_position = np.round(boundary_position, 2)

    n_loads = 1
    nodes_matrix = np.zeros((nely + 1, nelx + 1))

    if selected_boundary == 'left':
        nodes_matrix[int(boundary_position * nely):int(boundary_position * nely) + int(boundary_length), 0] = 1
    elif selected_boundary == 'right':
        nodes_matrix[int(boundary_position * nely):int(boundary_position * nely) + int(boundary_length), nelx] = 1
    elif selected_boundary == 'bottom':
        nodes_matrix[nely, int(boundary_position * nelx):int(boundary_position * nelx) + int(boundary_length)] = 1
    elif selected_boundary == 'top':
        nodes_matrix[0, int(boundary_position * nelx):int(boundary_position * nelx) + int(boundary_length)] = 1

    fixednode = np.argwhere(nodes_matrix.reshape((((nely + 1)) * ((nelx + 1))), order='F') == 1)
    fixeddofs = []

    if selected_boundary in ['left', 'right']:
        for i in range(len(fixednode)):
            fixeddofs.append(2 * fixednode[i])
            if selected_type == 'fully':
                fixeddofs.append((2 * fixednode[i]) + 1)
    elif selected_boundary in ['bottom', 'top']:
        for i in range(len(fixednode)):
            fixeddofs.append((2 * fixednode[i]) + 1)
            if selected_type == 'fully':
                fixeddofs.append(2 * fixednode[i])

    load_position = np.round(np.random.uniform(0, 1.0, size=n_loads), 2)
    while len(np.unique(load_position)) != n_loads:
        load_position = np.round(np.random.uniform(0, 0.99, size=n_loads), 2)

    support_orientation = calculate_support_orientation(selected_boundary)
    valid_load_orientations = filter_load_orientations(support_orientation)
    load_orientation = np.random.choice(valid_load_orientations, size=n_loads)
    magnitude_x = np.cos(np.radians(load_orientation))
    magnitude_y = np.sin(np.radians(load_orientation))

    for i in range(n_loads):
        if selected_boundary == 'left':
            nodes_matrix[int(load_position[i] * nely), nelx] = 2
        elif selected_boundary == 'right':
            nodes_matrix[int(load_position[i] * nely), 0] = 2
        elif selected_boundary == 'bottom':
            nodes_matrix[0, int(load_position[i] * nelx)] = 2
        elif selected_boundary == 'top':
            nodes_matrix[nely, int(load_position[i] * nelx)] = 2

    loadnode = np.argwhere(nodes_matrix.reshape((((nely + 1)) * ((nelx + 1))), order='F') == 2)
    loaddof_x = [2 * node for node in loadnode]
    loaddof_y = [(2 * node) + 1 for node in loadnode]

    boundary_length_norm = boundary_length / (nely if selected_boundary in ['left', 'right'] else nelx)

    out_dict = {
        'selected_boundary': boundaries.index(selected_boundary) / len(boundaries),
        'support_type': support_type.index(selected_type),
        'boundary_length': boundary_length_norm,
        'boundary_position': boundary_position,
        'n_loads': n_loads,
        'load_position': load_position,
        'load_orientation': np.array(load_orientation) / 360,
        'fixeddofs': fixeddofs,
        'loaddof_x': np.array(loaddof_x).flatten(),
        'loaddof_y': np.array(loaddof_y).flatten(),
        'magnitude_x': magnitude_x,
        'magnitude_y': magnitude_y,
        'volfrac': volume_fraction,
        'loadnode': loadnode,
        'fixednode': fixednode
    }

    if len(np.array(loaddof_x).flatten()) != len(magnitude_x):
        print('mismatch in rand_bc!!', loaddof_x, magnitude_x, load_position, n_loads)

    return dx, dy, nelx, nely, out_dict

def generate_prompt(conditions, dx, dy):
    dict = {
        'support': {
            'type': conditions['support_type'],
            'boundary': conditions['selected_boundary'],
            'length': conditions['boundary_length'],
            'position': conditions['boundary_position']
        },
        'domain': {
            'type': 'rectangular',
            'x_dimension': dx,
            'y_dimension': dy
        },
        'constraints': {
            'desired_volume_constraint': conditions['volfrac']
        }
    }

    for i in range(conditions['n_loads']):
        dict['load_' + str(i)] = {
            'type': 'point_load',
            'position': conditions['load_position'][i],
            'orientation': conditions['load_orientation'][i],
            'magnitude_x': conditions['magnitude_x'][i],
            'magnitude_y': conditions['magnitude_y'][i]
        }

    return str(dict)
