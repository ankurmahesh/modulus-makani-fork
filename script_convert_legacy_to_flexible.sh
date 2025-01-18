#!/bin/bash                                                             
                                                                        
                                                                        
for seed in 10                                                          
do                                                                      
    srun -N 1 --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-node=4 -u shifter --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint bash -c "source export_DDP_vars.sh; python3 -m makani.convert_legacy_to_flexible /pscratch/sd/a/amahesh/fcn_training/modulus-makani_runs-0.1.0gmd-fcndev_stats/sfno_linear_73chq_sc2_layers8_edim620_dt1h_wstgl2/v0.1.0-seed10/ /pscratch/sd/a/amahesh/earth2mip_prod_registry/sfno_linear_73chq_sc2_layers8_edim620_dt1h_wstgl2/v0.1.0-seed10/"
done   
