from dataclasses import dataclass
from typing import Any, Dict, List

from albumentations import Compose


@dataclass
class ConfigObject:
    type_object: object
    args: Dict[str, Any]


@dataclass
class ConfigPaths:
    logdir: str
    path_to_train_json: str
    path_to_test_json: str
    path_to_megadetector_json: str
    path_to_data_dir: str


@dataclass
class ConfigStage:
    stage_name: str

    num_epoch: int

    loss: ConfigObject
    sheduler: ConfigObject
    optimizer: ConfigObject

    pre_processing: Compose
    augmentations: Compose
    post_processing: Compose


@dataclass
class Config:
    paths: ConfigPaths
    stages: List[ConfigStage]

    model: ConfigObject

    is_fp16: bool
    is_verbose: bool
    is_minimize_valid_metric: bool

    device: str

    seed: int
    workers: int
    batch_size: int
    num_classes: int

    valid_size: float
