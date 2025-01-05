#!/bin/bash -l                                                          
#SBATCH --time=18:00:00                                                     
#SBATCH -C 'gpu&hbm80g'
#SBATCH --account=m1517
#SBATCH -q regular                                                      
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4                                             
#SBATCH --gpus-per-node=4                                               
#SBATCH --cpus-per-task=32
#SBATCH -J sc2_model_parallel
#SBATCH --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint
#SBATCH --module=gpu,nccl-2.18                                          
#SBATCH -o logs/0.1.0-sfno_linear_74chq_sc2-e620-%j.out
#SBATCH --mail-type=begin,end,fail                                      
#SBATCH --mail-user=amahesh@lbl.gov 
#SBATCH --array 112

export config_file=./config/sfnonet.yaml
export config='sfno_linear_74chq_sc2_layers8_edim620_wstgl2'
export run_num="v0.1.0-seed$SLURM_ARRAY_TASK_ID"

export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB
export HDF5_USE_FILE_LOCKING=FALSE

export MASTER_ADDR=$(hostname)
# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

export level=0
export multistep=1
export h_parallel_size=4
export w_parallel_size=1
export amp_mode="bf16"

export cmd="python -m makani.train --yaml_config=$config_file --config=$config --run_num=$run_num --amp_mode=$amp_mode  --checkpointing_level=$level --multistep_count=${multistep} --h_parallel_size=${h_parallel_size} --w_parallel_size=${w_parallel_size} --initialization_seed=$SLURM_ARRAY_TASK_ID" 

set -x

srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    ${cmd}
    "
