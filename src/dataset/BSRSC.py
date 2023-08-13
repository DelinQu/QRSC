# import sys; sys.path.append("/mnt/petrelfs/qudelin/PJLAB/RS/VRS-Transformer-dev")
import pathlib
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from src.dataset.transforms import ToTensor
import copy


class BSRSC(Dataset):
    def __init__(
        self,
        root_dir,
        seq_len=3,
        data_aug=ToTensor(),
        training=True,
    ):
        self.I_rs, self.I_gs = [], []
        self.seq_len = seq_len

        self.data_aug = copy.deepcopy(data_aug)
        self.training = training

        root_dir = pathlib.Path(root_dir)
        for seq_path in sorted(root_dir.iterdir()):
            seq_rs, seq_gs = sorted((seq_path / "RS").iterdir()), sorted((seq_path / "GS").iterdir())
            for i in range(len(seq_rs) - seq_len + 1):
                rs, gs = [], []
                for k in range(seq_len):
                    rs.append(seq_rs[i + k])
                    gs.append(seq_gs[i + k])

                self.I_rs.append(rs)
                self.I_gs.append(gs)

    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]

        I_rs = list()
        for i in range(self.seq_len):
            I_rs.append(Image.open(path_rs[i]))

        I_gs = Image.open(path_gs[self.seq_len // 2])

        paths = {
            "RS": str(path_rs[self.seq_len // 2]),
            "GS": str(path_gs[self.seq_len // 2]),
        }

        I_rs, I_gs = self.data_aug(I_gs.size, I_rs=I_rs, I_gs=I_gs)
        return I_rs, I_gs, [], paths


def get_data_loaders(
    train_dir="/mnt/petrelfs/share_data/qudelin/RS/BS-RSC/train",
    val_dir="/mnt/petrelfs/share_data/qudelin/RS/BS-RSC/val",
    test_dir="/mnt/petrelfs/share_data/qudelin/RS/BS-RSC/test",
    batch_size=2,
    shuffle=True,
    num_workers=1,
    seq_len=2,
    training=True,
    data_aug=ToTensor(),
):
    loader_args = {
        "batch_size": batch_size,
        # "shuffle": shuffle,
        "num_workers": num_workers,
    }

    if training:
        train_dataset = BSRSC(
            train_dir,
            seq_len=seq_len,
            data_aug=data_aug,
            training=training,
        )
        valid_dataset = BSRSC(val_dir, seq_len=seq_len, data_aug=data_aug, training=training)

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
            valid_sampler = DistributedSampler(valid_dataset, shuffle=shuffle)
            print("============== Use DistributedSampler =============")

        return DataLoader(train_dataset, shuffle=(train_sampler is None and shuffle), sampler=train_sampler, **loader_args), DataLoader(
            valid_dataset, shuffle=(valid_sampler is None and shuffle), sampler=valid_sampler, **loader_args
        )
    else:
        return DataLoader(
            BSRSC(
                test_dir,
                seq_len=seq_len,
                data_aug=data_aug,
                training=training,
            ),
            **loader_args
        )


if __name__ == "__main__":
    dataloader, *_ = get_data_loaders(training=True, data_aug=ToTensor())

    images, gt, *_ = next(iter(dataloader))
    print(len(images), images[0].shape, gt.shape)

    from torchvision.utils import save_image

    for i, rs in enumerate(images):
        save_image(rs, "rs_{:04d}.png".format(i))
    save_image(gt, "gs.png")

    for batch_idx, (data, target, *arg) in enumerate(dataloader):
        print(batch_idx, data[0].shape, target.shape)
