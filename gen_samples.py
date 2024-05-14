import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'       #Disactivate multiprocessing for numpy
from sogym.expert_generation import generate_dataset

generate_dataset(dataset_folder = "./dataset/topologies/cortex_may13",num_threads = 24, num_samples=10000)