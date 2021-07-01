import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import mobilenet_v2

from .base import ConfigPaths, Config, ConfigObject, ConfigStage


def get_config():
    num_classes = 511

    config_paths = ConfigPaths(
        logdir='/data/KAGGLE/iwildcam-2020-fgvc7/checkpoints/exp-0',
        path_to_train_json='/data/KAGGLE/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json',
        path_to_test_json='/data/KAGGLE/iwildcam-2020-fgvc7/iwildcam2020_test_information.json',
        path_to_megadetector_json='/data/KAGGLE/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json',
        path_to_data_dir='/data/KAGGLE/iwildcam-2020-fgvc7',
    )

    # PROCESSING`S
    pre_processing = albu.Compose((
        albu.Resize(512, 512),
    ))
    post_processing = albu.Compose((
        albu.Normalize(),
        ToTensorV2(),
    ))

    # AUGMENTATIONS
    augmentations = albu.Compose((
        albu.RGBShift(r_shift_limit=50,
                      g_shift_limit=50,
                      b_shift_limit=50,
                      p=.6),
        albu.IAAAffine(scale=1.02,
                       rotate=4,
                       shear=4,
                       order=0,
                       mode='constant',
                       p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.09,
                                      contrast_limit=0.09,
                                      p=0.5),
    ))

    stages_config = [
        ConfigStage(
            stage_name='train_0',

            num_epoch=1,

            loss=ConfigObject(
                type_object=BCEWithLogitsLoss,
                args={}
            ),
            optimizer=ConfigObject(
                type_object=Adam,
                args={'lr': 1e-4}
            ),
            sheduler=ConfigObject(
                type_object=CosineAnnealingLR,
                args={'T_max': 10, 'eta_min': 5e-5}
            ),

            pre_processing=pre_processing,
            augmentations=augmentations,
            post_processing=post_processing,

        )
    ]

    model = ConfigObject(
        type_object=mobilenet_v2,
        args={'num_classes': num_classes, 'pretrained': True}
    )

    config = Config(
        paths=config_paths,
        stages=stages_config,

        is_fp16=True,
        is_verbose=True,
        is_minimize_valid_metric=False,

        device='cuda:0',

        model=model,

        seed=666,
        workers=4,
        batch_size=10,
        num_classes=num_classes,

        valid_size=.2
    )

    return config
