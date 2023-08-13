from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.trainer import Trainer
from src.utils import get_logger, instantiate

# fix random seeds for reproducibility
SEED = 3047
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_worker(config):
    logger = get_logger("train")
    # setup data_loader instances
    # print("===== data_loader =====", config.data_loader)
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # logger.info(f"===== {config.data_loader._target_} =====")
    logger.info(model)
    logger.info(f"Trainable parameters: {sum([p.numel() for p in trainable_params])}")

    # get function handles of loss and metrics
    criterion = instantiate(config.loss)
    metrics = [instantiate(met, is_func=True) for met in config["metrics"]]

    # build optimizer, learning rate scheduler.
    # optimizer = instantiate(config.optimizer, model.parameters())
    optimizer = instantiate(config.optimizer, filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


def init_worker(rank, ngpus, working_dir, config):
    # initialize training config
    config = OmegaConf.create(config)
    config.local_rank = rank
    config.cwd = working_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:36183",
        world_size=ngpus,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    # start training processes
    train_worker(config)


@hydra.main(config_path="conf/", config_name="train", version_base="1.1")
def main(cfg):
    n_gpu = torch.cuda.device_count()
    assert n_gpu, "Can't find any GPU device on this machine."

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    if cfg.resume is not None:
        cfg.resume = hydra.utils.to_absolute_path(cfg.resume)

    if cfg.n_gpu > n_gpu:
        cfg.n_gpu = n_gpu
        print("===== available gpu number is {}, set n_gpu to {} =====".format(n_gpu, cfg.n_gpu))

    config = OmegaConf.to_yaml(cfg, resolve=True)
    torch.multiprocessing.spawn(init_worker, nprocs=cfg.n_gpu, args=(n_gpu, working_dir, config))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
