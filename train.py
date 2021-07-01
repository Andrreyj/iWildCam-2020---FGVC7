from catalyst.runners import SupervisedRunner
from catalyst import dl
from torch.utils.data import DataLoader

from config import Config, get_config
from config.base import ConfigStage
from src import WildCamDataset, load_data, prepare_env


def get_loaders(config: Config, stageconfig: ConfigStage):
    train_df, valid_df = load_data(config)

    train_dataloader = DataLoader(
        _get_dataset(train_df, config, stageconfig, True),
        batch_size=config.batch_size,
        num_workers=config.workers,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        _get_dataset(valid_df, config, stageconfig, False),
        batch_size=config.batch_size,
        num_workers=config.workers,
        shuffle=False
    )
    return {'train': train_dataloader, 'valid': valid_dataloader}


def _get_dataset(df, config: Config, stageconfig: ConfigStage, is_train):
    augmentations = None
    if is_train:
        augmentations = stageconfig.augmentations

    return WildCamDataset(
        df=df,
        num_labels=config.num_classes,
        root_path_data=config.paths.path_to_data_dir,
        pre_processing=stageconfig.post_processing,
        augmentations=augmentations,
        post_processing=stageconfig.post_processing,
    )


def get_callbacks(config: Config):
    callbacks = [
        dl.AccuracyCallback(input_key="logits",
                            target_key="target", topk_args=(1, 3, 5)),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="target",
            num_classes=config.num_classes,
        ),
    ]
    return callbacks


def main(config: Config):
    model = config.model.type_object(*config.model.args)
    runner = SupervisedRunner()

    for stageconf in config.stages:
        loss = stageconf.loss.type_object(*stageconf.loss.args)
        optimizer = stageconf.optimizer.type_object(
            model.parameters(), **stageconf.optimizer.args)
        sheduler = stageconf.sheduler.type_object(
            optimizer, **stageconf.sheduler.args)
        loaders = get_loaders(config, stageconf)
        callbacks = get_callbacks(config)

        continue
        runner.train(
            model=model,
            criterion=loss,
            optimizer=optimizer,
            sheduler=sheduler,
            loaders=loaders,
            seed=config.seed,
            num_epochs=stageconf.num_epoch,
            logdir=config.paths.logdir,
            fp16=config.is_fp16,
            minimize_valid_metric=config.is_minimize_valid_metric,
            verbose=config.is_verbose,
        )


if __name__ == '__main__':
    print('Hello world')
    config = get_config()

    prepare_env(config)

    main(config)
