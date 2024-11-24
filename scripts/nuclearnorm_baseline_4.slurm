#!/bin/bash
#SBATCH --job-name=nnm_4         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --constraint=cascade
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mlaprise@princeton.edu

module purge
export JULIA_NUM_THREADS=8
module load julia/1.10.5
cd /scratch/gpfs/mlaprise/socialcomputation/scripts
julia nuclearnorm_baseline.jl 4
