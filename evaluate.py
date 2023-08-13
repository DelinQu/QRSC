import logging
from pathlib import Path
import time
import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from src.utils import instantiate
from src.utils.time_util import AverageMeter
import pandas as pd

logger = logging.getLogger("evaluate")

@hydra.main(config_path="conf", config_name="evaluate", version_base="1.1")
def main(config):
    logger.info("Loading checkpoint: {} ...".format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)

    # setup data_loader instances
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(config.arch)

    # load trained weights
    state_dict = checkpoint["state_dict"]
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    metrics = [instantiate(met, is_func=True) for met in config.metrics]

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_losses = {}
    total_metrics = torch.zeros(len(metrics))
    metrics_data = {}
    for met in metrics:
        metrics_data[met.__name__] = list()

    timer = AverageMeter()
    with torch.no_grad():
        for data, target, mask, paths in tqdm(data_loader):
            B, _, _, _ = target.shape
            # TODO: make the data adapt to model and writer
            data = [img_.to(device) for img_ in data]
            target = target.to(device)
            mask = None if len(mask) == 0 else mask

            # * Timer
            start = time.time()
            _, _, output = model(data, config.gamma, config.tau)
            timer.update(time.time() - start)

            # TODO: metrics on test set.
            for k, metric in enumerate(metrics):
                score = metric(output, target, mask=mask) * B
                total_metrics[k] += score
                metrics_data[metric.__name__].append(score)

    n_samples = len(data_loader.sampler)

    # time
    log = {
        "dataset": config.data_loader._target_,
        "time per frame": timer.sum / n_samples,
    }

    # loss
    for k, v in total_losses.items():
        total_losses[k] = v / n_samples

    log.update(total_losses)

    # total metrics
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)})

    logger.info(OmegaConf.to_yaml(log))

    # item metric
    pd.DataFrame(metrics_data).to_csv("./metrics.csv")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
