#!/bin/bash
#SBATCH --account=e32998        # groups command on Quest to find out
#SBATCH --partition=gengpu      # leave as is
#SBATCH --gres=gpu:1            # number of GPUs (suggest 4-8)
#SBATCH --time=10:00            # suggest 12 hrs just in case
#SBATCH --nodes=1               # recommended is 1 node
#SBATCH --ntasks=1              # suggest 4
#SBATCH --mem=16G               # suggest 64GB
#SBATCH --output=test.out       # output file - rename
#SBATCH --error=test.err        # error file - rename
#SBATCH --job-name=test         # job name - rename
#SBATCH --mail-type=ALL         # send email at begin and end of job or failure
#SBATCH --mail-user=email       # your NU email address

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

python3 example.py