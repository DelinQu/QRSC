import torch
from torchvision.utils import make_grid
from .base import BaseTrainer
from src.utils import inf_loop, collect
from src.logger import BatchMetrics


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        args = ["total", *[n for n in criterion.names], *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(*args, postfix="/train", writer=self.writer)
        self.valid_metrics = BatchMetrics(*args, postfix="/valid", writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.data_loader.sampler.set_epoch(epoch)

        for batch_idx, (data, target, mask, _) in enumerate(self.data_loader):
            # TODO: make the data adapt to model and writer
            data = [img_.to(self.device) for img_ in data]
            target = target.to(self.device)
            mask = None if len(mask) == 0 else mask

            self.optimizer.zero_grad()
            out_ll, out_l, output = self.model(data, self.config.gamma, self.config.tau)
            losses = self.criterion(output, target)
            losses["total"].backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # TODO: update losses
            for name, loss in losses.items():
                losses[name] = collect(loss)
            self.train_metrics.update_scalars("loss", losses)

            # TODO: write image & metrics
            for met in self.metric_ftns:
                metric = collect(met(output, target, mask=mask))  # average metric between processes
                self.train_metrics.update(met.__name__, metric)

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # TODO: add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        self.train_metrics.write_epoch()
        self.valid_metrics.write_epoch()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, mask, _) in enumerate(self.valid_data_loader):
                # TODO: make the data adapt to model and writer
                data = [img_.to(self.device) for img_ in data]
                target = target.to(self.device)
                mask = None if len(mask) == 0 else mask

                _, _, output = self.model(data, self.config.gamma, self.config.tau)
                losses = self.criterion(output, target)

                # TODO: write image & metric log
                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                # self.writer.add_image(
                #     "valid/input",
                #     make_grid(data[len(data) // 2].cpu(), nrow=8, normalize=True),
                # )
                # self.writer.add_image(
                #     "valid/predict",
                #     make_grid(output.cpu(), nrow=8, normalize=True),
                # )

                # TODO: update losses
                for name, loss in losses.items():
                    losses[name] = collect(loss)
                self.valid_metrics.update_scalars("loss", losses)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, mask=mask))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        try:
            # ? epoch-based training the len of dataset?
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
            # if dist.is_initialized():
            #     current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)
