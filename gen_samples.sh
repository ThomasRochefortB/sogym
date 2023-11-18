#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --mail-user=thomas.rochefort.beaudoin@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH -o ./Report/output.%A_%a.out # STDOUT

module load gcc/9.3.0 opencv python scipy-stack
source ENV/bin/activate

python gen_samples.py
