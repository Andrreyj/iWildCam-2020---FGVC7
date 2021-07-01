import os
import random

import torch
import numpy as np

from config import Config, ConfigPaths


def prepare_env(conf: Config):
    _check_paths(conf.paths)
    _set_random_seed(conf.seed)
    _set_envoirment_variables()


def _set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_envoirment_variables():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def _check_paths(conf: ConfigPaths):
    if not os.path.isfile(conf.path_to_train_json):
        raise Exception(f'not foud path to dataset {conf.path_to_train_json=}')
    if not os.path.isfile(conf.path_to_test_json):
        raise Exception(f'not foud path to dataset {conf.path_to_test_json=}')
    if not os.path.isfile(conf.path_to_megadetector_json):
        raise Exception(
            f'not foud path to dataset {conf.path_to_megadetector_json=}')
    if not os.path.isdir(conf.path_to_data_dir):
        raise Exception(f'not foud path to data dir {conf.path_to_data_dir=}')
    if not os.path.isdir(conf.logdir):
        os.makedirs(conf.logdir, exist_ok=True)
