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

directory_path = "/home/thomas/Documents/scratch_thomas/GitHub/sogym_v2/dataset/topologies/unique_narval"
generate_expert_dataset(directory_path,env_kwargs,observation_type=observation_type, plot_terminated=False,num_permutations = num_permutations, file_fraction=1.0)