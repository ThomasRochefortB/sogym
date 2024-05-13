import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'       #Disactivate multiprocessing for numpy
from sogym.expert_generation import generate_dataset

generate_dataset(dataset_folder = "./dataset/topologies/holodeck_may12",num_threads = 32, num_samples=10000)