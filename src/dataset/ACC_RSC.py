"""RSC
load RSC images for metric, e.g, DSfM 
"""

# import sys; sys.path.append("/mnt/petrelfs/qudelin/PJLAB/RS/VRS-Transformer-dev")
import pathlib
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from src.dataset.transforms import ToTensor
import copy


class ACC(Dataset):
    def __init__(
        self,
        gs_dir,
        rsc_dir,
        seq_len=3,
        data_aug=ToTensor(),
        training=True,
    ):
        self.I_rs, self.I_gs = [], []
        self.seq_len = seq_len

        self.data_aug = copy.deepcopy(data_aug)
        self.training = training

        rsc_posix = pathlib.Path(rsc_dir)
        for seq_path in sorted(rsc_posix.iterdir()):
            for rsc in sorted(seq_path.iterdir()):
                rs, gs = str(rsc), str(rsc).replace(rsc_dir, gs_dir)
                self.I_rs.append(rs)
                self.I_gs.append(gs)

    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]

        I_rs = Image.open(path_rs)
        I_gs = Image.open(path_gs)

        paths = {
            "RS": path_rs,
            "GS": path_gs,
        }

        I_rs, I_gs = self.data_aug(I_gs.size, I_rs=I_rs, I_gs=I_gs)
        return I_rs, I_gs, [], paths


def get_data_loaders(
    gs_dir="",
    rsc_dir="",
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

    return DataLoader(
        ACC(
            gs_dir,
            rsc_dir,
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
