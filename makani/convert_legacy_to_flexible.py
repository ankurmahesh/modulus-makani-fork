# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import tempfile
from functools import partial

import pynvml
import torch
import torch.distributed as dist

from makani.models.model_package import (
    _load_static_data,
    MODEL_PACKAGE_CHECKPOINT_PATH,
    save_model_package,
    LocalPackage,
)
from makani.utils import logging_utils

from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani import Trainer
from makani.utils.YParams import ParamsBase


class CheckpointSaver(Trainer):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.
    """

    def __init__(self, params, world_rank):
        self.params = None
        self.world_rank = world_rank

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        params["resuming"] = False

        self.params = params
        self.model = model_registry.get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor
        self.iters = None
        self.optimizer = None
        self.epoch = None
        self.scheduler = None
        import logging
        self.logger = logging.getLogger()

        # print model
        if self.world_rank == 0:
            print(self.model)


def get_params(path):
    config = os.path.join(path, "config.json")
    return ParamsBase.from_json(config)


def save_checkpoint(path, output_path, rank, world_size, store_path, epoch):
    package = LocalPackage(path)
    params = get_params(path)
    if epoch is not None:
        params["checkpoint_path"] = params["checkpoint_path"].replace('.tar', '_epoch{}.tar'.format(args.epoch))
    #store = dist.FileStore(store_path, world_size)
    # setup distributed
    #dist.init_process_group(store=store, backend="nccl", rank=rank, world_size=world_size)
    # adjust checkpoint_path to be inside of ``path``. The checkpoint may not be in
    # the same location it was during training.
    checkpoint_template = os.path.basename(params.checkpoint_path)
    checkpoint_path = os.path.join(path, "training_checkpoints", checkpoint_template)
    params.log_to_wandb = False
    #with torch.cuda.device(dist.get_rank() % torch.cuda.device_count()):
    with torch.cuda.device(rank % torch.cuda.device_count()):
        _load_static_data(package, params)
        model_parallel_sizes = params.get("model_parallel_sizes", [1])
        model_parallel_names = params.get("model_parallel_names", ["model"])
        params.model_parallel_size = comm.init(model_parallel_sizes=model_parallel_sizes, model_parallel_names=model_parallel_names)
        saver = CheckpointSaver(params, world_rank=comm.get_world_rank())
        #saver.restore_checkpoint(checkpoint_path, checkpoint_mode=params["load_checkpoint"])
        saver.restore_checkpoint(checkpoint_path, checkpoint_mode='legacy')
        output_checkpoint_path = os.path.join(output_path, MODEL_PACKAGE_CHECKPOINT_PATH)
        if rank == 0:
            os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
        dist.barrier()
        saver.save_checkpoint(output_checkpoint_path, checkpoint_mode="flexible")

        params.experiment_dir = output_path
        if rank == 0:
            save_model_package(params)


help_str = """Convert legacy (checkpoint files per-rank) model packages to a single flexible
model package. 

This script should be run as a normal python script from an interactive session
with era5_wind installed.  Under the hood, it infers the number of needed ranks
to loads the original checkpoint and then spawns the needed processes.

Example::

    python3 convert_legacy_to_flexible.py /path_to_run_dir/sfno_linear_73chq_sc2_layers8_edim960_wstgl2/ngpu256_sp4/ package/
"""

if __name__ == "__main__":
    logging_utils.config_logger()
    parser = argparse.ArgumentParser(usage=help_str)
    parser.add_argument(
        "experiment_root",
        help="for example: sfno_linear_73chq_sc2_layers8_edim960_wstgl2/ngpu256_sp4",
    )
    parser.add_argument("output", help="Where to save the collected checkpoint.")
    parser.add_argument("--epoch", required=False, default=None, type=int, help='the epoch of the saved model to load')
    args = parser.parse_args()
    f = tempfile.mktemp()
    params = get_params(args.experiment_root)
    nproc = len(params.model_parallel_sizes)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print("nproc {}".format(nproc))
    print("world size {}".format(world_size))
    #assert nproc == world_size, "world size must equal the model parallel size as given by the config"
    save_checkpoint(args.experiment_root, args.output, rank, world_size, f, epoch=args.epoch)

    #torch.multiprocessing.spawn(
    #    partial(save_checkpoint, args.experiment_root, args.output),
    #    args=(nproc, f),
    #    nprocs=nproc,
    #)
