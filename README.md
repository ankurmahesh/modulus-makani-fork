*** Copyright Notice ***

Studies of Extreme Weather using Machine Learning and Climate Emulators
(HENS-SFNO) Copyright (c) 2024, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to receipt
of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


# HENS

This codebase is a fork of the earth2mip repository and the modulus-makani codebase, developed by NVIDIA.  It is used to run huge ensemble (HENS) weather forecasts with the SFNO architecture. It serves as the codebase for the following two papers:

"Huge Ensembles Part II: Properties of a Huge Ensemble of Hindcasts Generated with Spherical Fourier Neural Operators" (submitted to Geoscientific Model Development): https://arxiv.org/abs/2408.03100

"Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators" (submitted to Geoscientific Model development): https://arxiv.org/abs/2408.01581

The license and copyright for the code for these two projects are at:
earth2mip/license_lbl.txt
earth2mip/copyright_notice.txt

There are two primary codebases for HENS: modulus-makani (used for training the ML models) and earth2mip (used for inference, scoring, and analysis).  These two codebases are developed by NVIDIA, and we use forks of the codebases for this project.

The codebases for this are `earth2mip-fork` and `modulus-makani-fork` at the DOI listed in the code and data availability section in the HENS preprints. They are also made available on Github for reference and convenience.
The `earth2mip-fork` is also on Github at : https://github.com/ankurmahesh/earth2mip-fork/tree/HENS.  On Github, please be sure to use the "HENS" branch.
The `modulus-makani-fork` is also on Github at : https://github.com/ankurmahesh/modulus-makani-fork

The rest of this README has the following sections:
- Setting up the environment
- Training SFNO (`modulus-makani-fork`)
- Access to our trained SFNOs used for HENS
- Ensemble Inference with SFNO (`earth2mip-fork`)
- Scoring SFNO (`earth2mip-fork`)
- Analysis Scripts (`earth2mip-fork`)

If you would like to train SFNO, access our trained model weights, or run inference with SFNO, please read the respective sections.  These sections can be modified to according to various config files (YAML files for SFNO hyperparameters that you would like to train) for training and earth2mip configs for the inference.

If you would like to run our models for inference, please see the section on "Ensemble Inference with SFNO".  You should be able to download our trained models and adapt the earth2mip configs to run your own ensembles, either on the same or different initial dates, with different numbers of ensemble members, and with different variables saved.

Our analysis scripts are in the form of (1) postprocessing scripts to reshape the data and (2) Jupyter notebooks to perform the analysis and make the plots: this is included for reproducibility but often is specific to our filepaths.
## Setting up the environment

We use the docker to create an environment used for training and inference.  The docker environment can be created in `modulus-makani-fork`:
```
docker build modulus-makani-fork/docker/build.sh -t YOUR_TARGET_NAME:23.11
```

The docker image used in this work is on Dockerhub (https://hub.docker.com/r/amahesh19/modulus-makani/tags)

```
docker pull amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint
```

## Training SFNO: `modulus-makani-fork`

`modulus-makani` is the codebase used for training SFNO.  We use a fork of the NVIDIA modulus-makani v0.1.0.  We conduct training on Perlmutter at the NERSC on NVIDIA A100 GPUs: https://docs.nersc.gov/systems/perlmutter/architecture/

Our config of SFNO is `sfno_linear_74chq_sc2_layers8_edim620_wstgl2` at `modulus-makani-fork/config/sfnonet.yaml`. Our job to submit the training was run via

```sbatch modulus-makani-fork/mp_submit_pm.sh```

The above script runs the job for inference using spatial model parallelism (`h_parallel_size=4`, which means that the model is split across 4 ranks in the `h` spatial dimension).  Spatial model parallelism allows larger models to be trained, which may not otherwise fit in the GPU memory of 1 GPU. This means that the final trained model checkpoint is saved in 4 chunks.  For inference, we combine these 4 chunks into one, so that the model can be run for inference on a single rank.

We use the ```modulus-makani-fork/convert_legacy_to_flexible.py` Python file, which is called from the script:

```
bash modulus-makani-fork/script_convert_legacy_to_flexible.sh
```

## Access to our trained SFNOs used for HENS

We open-source the learned weights of our models at 3 locations: (1) the DataDryad DOI listed in the paper, (2) (https://huggingface.co/datasets/maheshankur10/hens/tree/main) https://doi.org/10.57967/hf/4200, and (3) https://portal.nersc.gov/cfs/m4416/hens/earth2mip\_prod\_registry/.  Each model is trained with a different initial seed: therefore, the name of each trained model end with "seed[SEED]," where SEED corresponds to the Pytorch seed used for the weight initialization. Each model also includes a `config.json` file which specifies the configuration parameters used for training each SFNO checkpoint. Each model package includes the learned weights of the model and additional information necessary to run the model (e.g. the input to the model is normalized by the data in 'global_means.npy' and 'global_stds.npy').

As the data is hosted on the NERSC high-performance computing facility, it may occasionally be down for maintenance.  If the link above does not work, you can check https://www.nersc.gov/live-status/motd/ to see if the data portal is up.  In cases of downtime, the model checkpoints may also be downloaded at https://huggingface.co/datasets/maheshankur10/hens/tree/main (https://doi.org/10.57967/hf/4200) or the DataDryad repository listed in the Code and Data Availability Section of the HENS preprints.

## Ensemble Inference with SFNO: `earth2mip-fork`

### ERA5 data for inference

We use the files at `modulus-makani-fork/data_process/` to create the ERA5 mirror used for training and inference.

### Running the model for inference

#### Roadmap and important files

`earth2mip-fork` is used to run the ensemble weather forecasts.  In the HENS manuscripts, we use two methods to create an ensemble: bred vector initial condition perturbations (to represent initial condition uncertainty) and multiple checkpoints trained from scratch (to represent model uncertainty). Particular files of interest are our implementation of bred vectors for initial condition perturbations:

'earth2mip-fork/earth2mip/ensemble_utils.py'

Additionally, ensemble inference with the multiple checkpoints is at 

'earth2mip-fork/earth2mip/inference_ensemble.py'

#### Scripts and configs to run the ensemble

In order to run the ensemble, some paths to some dependent data files must be set as environment variables.  The script to set these paths is at earth2mip-fork/set_74ch_vars.sh.  The environment variables can be changed according to the location of the data dependencies on your machine. `prod_heat_index_lookup.zarr` is a table used for calculating the heat index, `percentile_95.tar` is a tar of the 95th percentile calculated for each hour of day and each month of ERA5 used for thresholding in the calculation of extreme statistics, and `percentile_99.tar` is the 99th percentile from ERA5.  `d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc` includes the data that stores the amplitudes used for the bred vector initial condition perturbation method.

This data for these dependent data files is available at https://huggingface.co/datasets/maheshankur10/hens/tree/main and at https://portal.nersc.gov/cfs/m4416/hens/.  You'll have to download this data to your local machine and change the paths `set_74ch_vars.sh` to point to this data.

The general documentation for earth2mip is included below.  For running ensembles, we recommend setting up earth2mip inference_ensemble to work on your local compute environment (ideally using the general earth2mip documentation).  Then, you can make modifications to the earth2mip config json to run ensembles with our trained model. In particular, earth2mip requires defining a config file which sets the parameters of the ensemble.  The config files used for generating huge ensemble forecasts in summer 2023 are at 

'earth2mip-fork/summer23_configs/*'

If you would like to run an ensemble with our models, you can adapt these configs to suit your local directory (e.g. change the save paths) and also to run ensembles that save different variable subsets and use different days as initial conditions.  See the earth2mip documentation for more information about specifying a config for ensemble inference.

The submit script for the huge ensemble is 

'sbatch earth2mip-fork/submit_HENS_summer23.sh' . This script creates a huge (7424 member ensemble) using `earth2mip/inference_ensemble.py` mentioned above

### Hyperparameter Tuning (Figures 2 and 3)

We perform minimal hyperparameter tuning (exploring a small model,  medium model, and large model).  We analyze the spectra of each model size.  And we assess the lagged ensemble performance of each model size.  This allows to assess ensemble performance without having to train multiple checkpoints of each model. You can read more about lagged ensembles in the preprints and at https://arxiv.org/pdf/2401.15305.

The code to view the results of the tuning is at `earth2mip-fork/earth2mip/scripts/spectra/SFNO_Hyperparameter_Spectra.ipynb` (HENS Part 1 Figure 2)

The code to view the results of resampling the number of checkpoints, to determine how many checkpoints is enough is at `earth2mip-fork/scripts/num_checkpoints/MakeFigure-Number_of_Checkpoints.ipynb`

### Scoring the ensemble: overall diagnostics

`earth2mip-fork/earth2mip/score_ensemble_outputs.py` and `earth2mip-fork/earth2mip/time_collection.py` is the file used to score the ensemble against the ERA5 dataset.  We score the dataset using 2018, 2020, and 2023.  This file used `inference_ensemble.py` to generate an ensemble and then uses the earth2mip scoring utils to score the ensemble.  The overall diagnostics include spread-error ratio, CPRS, and ensemble mean RMSE (Part 1: Figures 6-9).

We calculate the overall diagnostics across the entire year, using 730 initial times (00:00 UTC and 12:00 UTC for each day of the validation year).  This diagnostics calculation workflow is through `time_collection.py`, in which the ensemble is run, the output is saved to disk, and the scores are calculated.  If `save_ensemble` is set to True in the time_collection JSON below, then the ensemble is saved as a zarr file; otherwise, if it is false, then the ensemble is deleted.  There is a general template file for this time collection job: `earth2mip-fork/time_collection_config.json`: this config includes the parameters of the ensemble.  The `earth2mip/make_job.py` file converts this template into a config that can be used for the time_collection.

We have our scoring ensemble script here:

```
python earth2mip/make_job.py multicheckpoint time_collection_config.json /PATH/TO/SAVE/OUTPUT/
sbatch original_submit_time_collection.sh
```

### Scoring the ensemble: spectral diagnostics

We include diagnostics of the spectral properties of each member and the ensemble mean (Part 1: Figures 10-12).  The code for deterministic and perturbed spectra is at `earth2mip-fork/earth2mip/scripts/spectra/MakeFigure_Deterministic_Spectral_Analysis.ipynb` . The code for the ensemble spectra is at `earth2mip-fork/earth2mip/scripts/spectra/MakeFigure-EnsembleMeanSpectralAnalysis.ipynb`.

### Scoring the ensemble: extreme diagnostics

The diagnostics on extremes are located in `earth2mip/extreme_scoring_utils.py` and `earth2mip-fork/earth2mip/score_extreme_diagnostics.py`.  These include diagnostics like reliability diagrams, threshold-weighted CRPS, and ROC curves, and Extreme Forecast Index (Part 1: Figures 13-15).

####Extreme Forecast Index Prerequistites

IFS EFI values are directly downloaded from ECMWF (we do not calculate the IFS EFI values).  The SFNO-BVMC requires a model climatology and percentiles on this climatology.  

```
python earth2mip/make_job.py multicheckpoint time_collection_config_mclimate.json /PATH/TO/SAVE/MCLIMATE/OUTPUT/
sbatch submit_mclimate.sh

# After the job is completed, the percentiles are calculated with
sbatch earth2mip/scripts/cdf/mclimate-percentiles-slurmarray.sh
```
#### Calculating the extreme diagnostics

The IFS ensemble is downloaded from TIGGE. The SFNO-BVMC ensemble is generated using the time_collection workflow above (and save_ensemble=True in the time_collection_config).

The scripts that we used to calculate the extreme diagnostics are at

```
bash efi_extreme_scoring_ifs.sh #for IFS
bash efi_score_prerun_sfno.sh #for SFNO-BVMC
```

# Analysis of Huge Ensemble Run of Summer 2023 (figures in HENS Part 2)

We conducted a huge ensemble run with 7,424 ensemble members, initialized each day of summer 2023 and run forward in time for 15 days. In total, this creates almost 25 petabytes of data across all variables.  We save a subset of the variables (accounting for 2 petabytes).  We analyze the properties of this large dataset in HENS Part 2.

## Figures 2,4

The analysis code for Figures 2,4 is available at hens_gmd_reproducibility.zip

## Figure 5: Demo

We run the huge ensemble using using the submit_HENS_summer23.sh script above. We generate the demo for Kansas City using the code at `earth2mip-fork/earth2mip/scripts/demo/kc.py` and `earth2mip-fork/earth2mip/scripts/demo/KansasCity.ipynb`.  This code extracts the forecasts for the Kansas City heatwave initialized at the 10 initial times leading up to the heatwave, and it visualizes the 2D distribution t2m and d2m of one initialized forecast of the heatwave.

## Figures 3,6-12: Calculation of statistics on HENS Summer 2023

We calculated gain at each grid cell with `earth2mip-fork/earth2mip/scripts/HENS_stats/paralleltrials_gain_gridcell.py` and `earth2mip-fork/earth2mip/scripts/HENS_stats/submit_paralleltrials_gain_gridcell.sh` (Figure 3)

In order to calculate statistics on the HENS run for summer 2023, we first had to perform a matrix transpose, to change the way the data was stored in memory.  This transpose enables taking reductions (e.g. mean, min, max) over the huge ensemble dimension (7424).  See HENS Part II Appendix A for more information.

The code to perform this transpose is at `earth2mip-fork/earth2mip/scripts/h5_convert/h5_convert.py`.  We ran the job with

```
sbatch earth2mip-fork/earth2mip/scripts/h5_convert/alldates_submit.sh
```

After the data was transformed, we calculated statistics on the data with `earth2mip-fork/earth2mip/scripts/HENS_stats/reduce.py`.  These statistics tell us the ensemble mean, min, max, variance, and ensemble mean RMSE.  Additionally, we also calculated statistics on bootstrapped ensembles of different sizes (50, 100, 500, 1000, 5000, and 7424 members), including satisfactory ensemble size (ensemble size where 95% of bootstrapped ensembles include the true value), confidence interval of extreme 2m temperature, mean probability of extreme t2m, and RMSE of the best member (with a confidence interval).

```
sbatch earth2mip-fork/earth2mip/scripts/HENS_stats/submit_reduce.sh
```

We calculated Figure 6: number of ensemble members that exceed ERA5 (binned for different sigma values of ERA5) with `earth2mip-fork/earth2mip/scripts/HENS_stats/number_of_samples.ipynb`

We calculated Figure 7: minimum RMSE at `earth2mip-fork/earth2mip/scripts/HENS_stats/Minimum_RMSE.ipynb`

 We calculated the twCRPS, owCRPS and CRPS figures (Figure 8) at `earth2mip-fork/earth2mip/scripts/HENS_stats/Calculate_HENS_CRPS_related.ipynb`. We calculated the twCRPS, owCRPS, and CRPS walkthrough figures (Figure 9) at `earth2mip-fork/earth2mip/scripts/HENS_stats/Create_HENSpartII_walkthrough_figures.ipynb`. 

We calculated the width of the confidence intervals with `earth2mip-fork/earth2mip/scripts/HENS_stats/Confidence_Intervals.ipynb` (Figure 10), which uses the output of the reduce.py file above.

We calculated Figure 11: outlier statistic with `earth2mip-fork/earth2mip/scripts/HENS_stats/Necessary_ensemble_members.ipynb`

We calculated Figure 12 (for the cases where the true value is greater than the IFS ensemble max, what is the HENS max) with `earth2mip-fork/earth2mip/scripts/HENS_stats/HENSmax_IFSmax_ERA5ZScore.ipynb`

# Makani: Massively parallel training of machine-learning based weather and climate models

[**Overview**](#overview) | [**Getting started**](#getting-started) | [**More information**](#more-about-makani) | [**Known issues**](#known-issues) | [**Contributing**](#contributing) | [**Further reading**](#further-reading) | [**References**](#references)

[![tests](https://github.com/NVIDIA/makani/actions/workflows/tests.yml/badge.svg)](https://github.com/NVIDIA/makani/actions/workflows/tests.yml)

Makani (the Hawaiian word for wind 🍃🌺) is an experimental library designed to enable the research and development of machine-learning based weather and climate models in PyTorch. Makani is used for ongoing research. Stable features are regularly ported to the [NVIDIA Modulus](https://developer.nvidia.com/modulus) framework, a framework used for training Physics-ML models in Science and Engineering.

<div align="center">
<img src="https://github.com/NVIDIA/makani/blob/main/images/sfno_rollout.gif"  height="388px">
</div>

## Overview

Makani was started by engineers and researchers at NVIDIA and NERSC to train [FourCastNet](https://github.com/NVlabs/FourCastNet), a deep-learning based weather prediction model.

Makani is a research code built for massively parallel training of weather and climate prediction models on 100+ GPUs and to enable the development of the next generation of weather and climate models. Among others, Makani was used to train [Spherical Fourier Neural Operators (SFNO)](https://developer.nvidia.com/blog/modeling-earths-atmosphere-with-spherical-fourier-neural-operators/) [1] and [Adaptive Fourier Neural Operators (AFNO)](https://arxiv.org/abs/2111.13587) [2] on the ERA5 dataset. Makani is written in [PyTorch](https://pytorch.org) and supports various forms of model- and data-parallelism, asynchronous loading of data, unpredicted channels, autoregressive training and much more.

## Getting started

Makani can be installed by running

```bash
git clone git@github.com:NVIDIA/makani.git
cd makani
pip install -e .
```

### Training:

Training is launched by calling `train.py` and passing it the necessary CLI arguments to specify the configuration file `--yaml_config` and he configuration target `--config`:

```bash
mpirun -np 8 --allow-run-as-root python -u makani.train --yaml_config="config/sfnonet.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2"
```

:warning: **architectures with complex-valued weights** will currently fail. See  [Known issues](#known-issues) for more information.

Makani supports various optimization to fit large models ino GPU memory and enable computationally efficient training. An overview of these features and corresponding CLI arguments is provided in the following table:

| Feature                   | CLI argument                                  | options                      |
|---------------------------|-----------------------------------------------|------------------------------|
| Automatic Mixed Precision | `--amp_mode`                                  | `none`, `fp16`, `bf16`       |
| Just-in-time compilation  | `--jit_mode`                                  | `none`, `script`, `inductor` |
| CUDA graphs               | `--cuda_graph_mode`                           | `none`, `fwdbwd`, `step`     |
| Activation checkpointing  | `--checkpointing_level`                       | 0,1,2,3                      |
| Data parallelism          | `--batch_size`                                | 1,2,3,...                    |
| Channel parallelism       | `--fin_parallel_size`, `--fout_parallel_size` | 1,2,3,...                    |
| Spatial model parallelism | `--h_parallel_size`, `--w_parallel_size`      | 1,2,3,...                    |
| Multistep training        | `--multistep_count`                           | 1,2,3,...                    |

Especially larger models are enabled by using a mix of these techniques. Spatial model parallelism splits both the model and the data onto multiple GPUs, thus reducing both the memory footprint of the model and the load on the IO as each rank only needs to read a fraction of the data. A typical "large" training run of SFNO can be launched by running

```bash
mpirun -np 256 --allow-run-as-root python -u makani.train --amp_mode=bf16 --cuda_graph_mode=fwdbwd --multistep_count=1 --run_num="ngpu256_sp4" --yaml_config="config/sfnonet.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --h_parallel_size=4 --w_parallel_size=1 --batch_size=64
```
Here we train the model on 256 GPUs, split horizontally across 4 ranks with a batch size of 64, which amounts to a local batch size of 1/4. Memory requirements are further reduced by the use of `bf16` automatic mixed precision.

### Inference:

In a similar fashion to training, inference can be called from the CLI by calling `inference.py` and handled by `inferencer.py`. To launch inference on the out-of-sample dataset, we can call:

```bash
mpirun -np 256 --allow-run-as-root python -u makani.inference --amp_mode=bf16 --cuda_graph_mode=fwdbwd --multistep_count=1 --run_num="ngpu256_sp4" --yaml_config="config/sfnonet.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --h_parallel_size=4 --w_parallel_size=1 --batch_size=64
```

By default, the inference script will perform inference on the out-of-sample dataset specified 

## More about Makani

### Project structure

The project is structured as follows:

```
makani
├── ...
├── config                      # configuration files, also known as recipes
├── data_process                # data pre-processing such as computation of statistics
├── datasets                    # dataset utility scripts
├── docker                      # scripts for building a docker image for training
├── makani                      # Main directory containing the package
│   ├── inference               # contains the inferencer
│   ├── mpu                     # utilities for model parallelism
│   ├── networks                # networks, contains definitions of various ML models
│   ├── third_party/climt       # third party modules
│   │   └── zenith_angle.py     # computation of zenith angle
│   ├── utils                   # utilities
│   │   ├── dataloaders         # contains various dataloaders
│   │   ├── metrics             # metrics folder contains routines for scoring and benchmarking.
│   │   ├── ...
│   │   ├── comm.py             # comms module for orthogonal communicator infrastructure
│   │   ├── dataloader.py       # dataloader interface
│   │   ├── metric.py           # centralized metrics handler
│   │   ├── trainer_profile.py  # copy of trainer.py used for profiling
│   │   └── trainer.py          # main file for handling training
│   ├── ...
│   ├── inference.py            # CLI script for launching inference
│   ├── train.py                # CLI script for launching training
├── tests                       # test files
└── README.md                   # this file
```

### Model and Training configuration
Model training in Makani is specified through the use of `.yaml` files located in the `config` folder. The corresponding models are located in `networks` and registered in the `get_model` routine in `networks/models.py`. The following table lists the most important configuration options.

| Configuration Key         | Description                                             | Options                                                 |
|---------------------------|---------------------------------------------------------|---------------------------------------------------------|
| `nettype`                 | Network architecture.                                   | `SFNO`, `FNO`, `AFNO`, `ViT`                            |
| `loss`                    | Loss function.                                          | `l2`, `geometric l2`, ...                               |
| `optimizer`               | Optimizer to be used.                                   | `Adam`, `AdamW`                                         |
| `lr`                      | Initial learning rate.                                  | float > 0.0                                             |
| `batch_size`              | Batch size.                                             | integer > 0                                             |
| `max_epochs`              | Number of epochs to train for                           | integer                                                 |
| `scheduler`               | Learning rate scheduler to be used.                     | `None`, `CosineAnnealing`, `ReduceLROnPlateau`, `StepLR`|
| `lr_warmup_steps`         | Number of warmup steps for the learning rate scheduler. | integer >= 0                                            |
| `weight_decay`            | Weight decay.                                           | float                                                   |
| `train_data_path`         | Directory path which contains the training data.        | string                                                  |
| `test_data_path`          | Network architecture.                                   | string                                                  |
| `exp_dir`                 | Directory path for ouputs such as model checkpoints.    | string                                                  |
| `metadata_json_path`      | Path to the metadata file `data.json`.                  | string                                                  |
| `channel_names`           | Channels to be used for training.                       | List[string]                                            |


For a more comprehensive overview, we suggest looking into existing `.yaml` configurations. More details about the available configurations can be found in [this file](config/README.md).

### Training data
Makani expects the training/test data in HDF5 format, where each file contains the data for an entire year. The dataloaders in Makani will then load the input `inp` and the target `tar`, which correspond to the state of the atmosphere at a given point in time and at a later time for the target. The time difference between input and target is determined by the parameter `dt`, which determines how many steps the two are apart. The physical time difference is determined by the temporal resolution `dhours` of the dataset.

Makani requires a metadata file named `data.json`, which describes important properties of the dataset such as the HDF5 variable name that contains the data. Another example are channels to load in the dataloader, which arespecified via channel names. The metadata file has the following structure:

```json
{
    "dataset_name": "give this dataset a name",     # name of the dataset
    "attrs": {                                      # optional attributes, can contain anything you want
        "decription": "description of the dataset",
        "location": "location of your dataset"
    },
    "h5_path": "fields",                            # variable name of the data inside the hdf5 file
    "dims": ["time", "channel", "lat", "lon"],      # dimensions of fields contained in the dataset
    "dhours": 6,                                    # temporal resolution in hours
    "coord": {                                      # coordinates and channel descriptions
        "grid_type": "equiangular",                 # type of grid used in dataset: currently suppported choices are 'equiangular' and 'legendre-gauss'
        "lat": [0.0, 0.1, ...],                     # latitudinal grid coordinates
        "lon": [0.0, 0.1, ...],                     # longitudinal grid coordinates
        "channel": ["t2m", "u10", "v10", ...]       # names of the channels contained in the dataset
    }
}
```

The ERA5 dataset can be downloaded [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

### Model packages

By default, Makani will save out a model package when training starts. Model packages allow easily contain all the necessary data to run the model. This includes statistics used to normalize inputs and outputs, unpredicted static channels and even the code which appends celestial features such as the cosine of the solar zenith angle. Read more about model packages [here](networks/Readme.md).

## Known Issues

:warning: **architectures with complex-valued weights**: Training some architectures with complex-valued weights requires yet to be released patches to PyTorch. A hotfix that addresses these issues is available in the `makani/third_party/torch` folder. Overwriting the corresponding files in the PyTorch installation will resolve these issues.

## Contributing

Thanks for your interest in contributing. There are many ways to contribute to this project.

- If you find a bug, let us know and open an issue. Even better, if you feel like fixing it and making a pull-request, we are incredibly grateful for that. 🙏
- If you feel like adding a feature, we encourage you to discuss it with us first, so we can guide you on how to best achieve it.

While this is a research project, we aim to have functional unit tests with decent coverage. We kindly ask you to implement unit tests if you add a new feature and it can be tested.

## Further reading

- [Modulus](https://developer.nvidia.com/modulus), NVIDIA's library for physics-ML
- [NVIDIA blog article](https://developer.nvidia.com/blog/modeling-earths-atmosphere-with-spherical-fourier-neural-operators/) on Spherical Fourier Neural Operators for ML-based weather prediction
- [torch-harmonics](https://github.com/NVIDIA/torch-harmonics), a library for differentiable Spherical Harmonics in PyTorch
- [ECMWF ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
- [SFNO-based forecasts deployed by ECMWF](https://charts.ecmwf.int/products/fourcast_medium-mslp-wind850)
- [Apex](https://github.com/NVIDIA/apex), tools for easier mixed precision
- [Dali](https://developer.nvidia.com/dali), NVIDIA data loading library
- [earth2mip](https://github.com/NVIDIA/earth2mip), a library for intercomparing DL based weather models

## Authors

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-d@2x.png"  height="120px"><img src="https://www.nersc.gov/assets/Logos/NERSClogocolor.png"  height="120px">

The code was developed by Thorsten Kurth, Boris Bonev, Jean Kossaifi, Animashree Anandkumar, Kamyar Azizzadenesheli, Noah Brenowitz, Ashesh Chattopadhyay, Yair Cohen, David Hall, Peter Harrington, Pedram Hassanzadeh, Christian Hundt, Alexey Kamenev, Karthik Kashinath, Zongyi Li, Morteza Mardani, Jaideep Pathak, Mike Pritchard, David Pruitt, Sanjeev Raja, Shashank Subramanian.


## References

<a id="#sfno_paper">[1]</a> 
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
arXiv 2306.0383, 2023.

<a id="1">[2]</a> 
Pathak J., Subramanian S., Harrington P., Raja S., Chattopadhyay A., Mardani M., Kurth T., Hall D., Li Z., Azizzadenesheli K., Hassanzadeh P., Kashinath K., Anandkumar A.;
FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators;
arXiv 2202.11214, 2022.

## Citation

If you use this package, please cite

```bibtex
@InProceedings{bonev2023sfno,
    title={Spherical {F}ourier Neural Operators: Learning Stable Dynamics on the Sphere},
    author={Bonev, Boris and Kurth, Thorsten and Hundt, Christian and Pathak, Jaideep and Baust, Maximilian and Kashinath, Karthik and Anandkumar, Anima},
    booktitle={Proceedings of the 40th International Conference on Machine Learning},
    pages={2806--2823},
    year={2023},
    volume={202},
    series={Proceedings of Machine Learning Research},
    month={23--29 Jul},
    publisher={PMLR},
}

@article{pathak2022fourcastnet,
    title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
    author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
    journal={arXiv preprint arXiv:2202.11214},
    year={2022}
}
```
