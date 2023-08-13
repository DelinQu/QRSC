# import sys; sys.path.append("/mnt/petrelfs/qudelin/PJLAB/RS/VRS-Transformer")
import os
import os.path as osp
import torch.distributed as dist
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from src.dataset.transforms import ToTensor
import copy


class Fastec(Dataset):
    def __init__(
        self,
        root_dir,
        seq_len=3,
        load_optiflow=True,
        load_middle_gs=False,
        data_aug=ToTensor(),
        training=True,
    ):
        self.I_rs, self.I_gs, self.I_gs_f = [], [], []
        self.optiflow, self.flow_m = [], []
        self.seq_len = seq_len
        self.load_optiflow = load_optiflow
        self.data_aug = copy.deepcopy(data_aug)
        self.training = training

        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != "meta.log":
                    continue

                #! read in seq of images
                for i in range(34 - self.seq_len + 1):
                    if not osp.exists(osp.join(seq_path, str(i).zfill(3) + "_rolling.png")):
                        continue

                    seq_Irs, seq_Igs, seq_Igs_f = [], [], []
                    seq_optiflow, seq_flow_m = [], []

                    seq_Irs.append(osp.join(seq_path, str(i).zfill(3) + "_rolling.png"))
                    if load_middle_gs:
                        seq_Igs.append(osp.join(seq_path, str(i).zfill(3) + "_global_middle.png"))
                    else:
                        seq_Igs.append(osp.join(seq_path, str(i).zfill(3) + "_global_first.png"))
                    seq_flow_m.append(osp.join(seq_path, str(i).zfill(3) + "_flow_raft_m.flo"))

                    seq_optiflow.append(osp.join(seq_path, str(i).zfill(3) + "_flow_raft.flo"))

                    # ! read another images in seq_len.
                    for j in range(1, seq_len):
                        seq_Irs.append(osp.join(seq_path, str(i + j).zfill(3) + "_rolling.png"))
                        if load_middle_gs:
                            seq_Igs.append(osp.join(seq_path, str(i + j).zfill(3) + "_global_middle.png"))
                        else:
                            seq_Igs.append(osp.join(seq_path, str(i + j).zfill(3) + "_global_first.png"))
                        seq_flow_m.append(osp.join(seq_path, str(i + j).zfill(3) + "_flow_raft_m.flo"))
                        seq_optiflow.append(osp.join(seq_path, str(i + j).zfill(3) + "_flow_raft.flo"))

                    if osp.exists(seq_optiflow[0]) or not osp.exists(seq_optiflow[1]):
                        seq_optiflow[1] = seq_optiflow[0]

                    if not osp.exists(seq_Irs[-1]):
                        break

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())
                    self.optiflow.append(seq_optiflow.copy())
                    self.flow_m.append(seq_flow_m.copy())

    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]

        # ! Read images and Data augmentation from seq
        I_rs = list()
        seed = random.randint(0, 2**32)
        for i in range(self.seq_len):
            random.seed(seed)
            I_rs.append(Image.open(path_rs[i]))

        I_gs = Image.open(path_gs[self.seq_len // 2])

        paths = {
            "RS": str(path_rs[self.seq_len // 2]),
            "GS": str(path_gs[self.seq_len // 2]),
        }

        I_rs, I_gs = self.data_aug(I_gs.size, I_rs=I_rs, I_gs=I_gs)
        return I_rs, I_gs, [], paths


def get_data_loaders(
    train_dir="/mnt/lustre/qudelin/DATA/RS/Fastec/train",
    val_dir="/mnt/lustre/qudelin/DATA/RS/Fastec/val",
    test_dir="/mnt/lustre/qudelin/DATA/RS/Fastec/test",
    batch_size=2,
    shuffle=True,
    num_workers=1,
    seq_len=3,
    training=True,
    load_middle_gs=False,
    data_aug=ToTensor(),
):
    loader_args = {
        "batch_size": batch_size,
        # "shuffle": shuffle,
        "num_workers": num_workers,
    }

    if training:
        train_dataset = Fastec(train_dir, seq_len=seq_len, data_aug=data_aug, training=training, load_middle_gs=load_middle_gs)
        valid_dataset = Fastec(val_dir, seq_len=seq_len, data_aug=data_aug, training=training, load_middle_gs=load_middle_gs)

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
            valid_sampler = DistributedSampler(valid_dataset, shuffle=shuffle)
            # print("============== Use DistributedSampler =============")

        return DataLoader(train_dataset, shuffle=(train_sampler is None and shuffle), sampler=train_sampler, **loader_args), DataLoader(
            valid_dataset, shuffle=(valid_sampler is None and shuffle), sampler=valid_sampler, **loader_args
        )
    else:
        return DataLoader(Fastec(test_dir, seq_len=seq_len, data_aug=data_aug, training=training, load_middle_gs=load_middle_gs), **loader_args)


if __name__ == "__main__":
    dataloader, *_ = get_data_loaders(training=True, data_aug=ToTensor())

    images, gt, *_ = next(iter(dataloader))
    print(len(images), images[0].shape, gt.shape)

    from torchvision.utils import save_image

    for i, rs in enumerate(images):
        save_image(rs, "rs_{:04d}.png".format(i))
    save_image(gt, "gs.png")

    for batch_idx, (data, target, *arg) in enumerate(dataloader):
        # print(batch_idx, data[0].shape, target.shape, arg)
        print(batch_idx, data[0].shape, target.shape)
