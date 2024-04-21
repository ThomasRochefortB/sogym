from sogym.expert_generation import generate_expert_dataset
import pickle

# Specify the number of permutations to generate
num_permutations = None
observation_type = "topopt_game"

# Specify the environment configuration (optional)
env_kwargs = {
    'mode': 'train',
    'observation_type': observation_type,
    'vol_constraint_type': 'hard',
    'seed': 42,
    'resolution' : 50,
    'check_connectivity':True
}

directory_path = "/home/thomas/Documents/scratch_thomas/GitHub/sogym_v2/dataset/topologies_narval"
expert_observations, expert_actions = generate_expert_dataset(directory_path,env_kwargs, plot_terminated=False,num_permutations = num_permutations, file_fraction=1.0)
# Save the dataset

# Determine the permutation part of the filename
if num_permutations is None:
    perm_str = 'noperm'
else:
    perm_str = f"{num_permutations}perm"
filename = f"dataset/expert/expert_dataset_{observation_type}_{perm_str}.pkl"

# Save the data using pickle
with open(filename, 'wb') as f:
    pickle.dump({'expert_observations': expert_observations, 'expert_actions': expert_actions}, f, protocol=4)
print(len(expert_observations))