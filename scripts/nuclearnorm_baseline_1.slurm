#!/bin/bash
#SBATCH --job-name=nnm_1         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=01:02:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mlaprise@princeton.edu

module purge
module load julia/1.10.5
cd /scratch/gpfs/mlaprise/socialcomputation
julia import Pkg; Pkg.activate(".")
julia nuclearnorm_baseline.jl 1
