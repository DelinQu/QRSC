import os
import os.path as osp
import torch.distributed as dist
import sys

sys.path.append("/mnt/petrelfs/qudelin/PJLAB/QRST/QRST-dev")
import copy
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from src.dataset.transforms import RandomCrop, ToTensor


class Carla(Dataset):
    """
    Creates a pair of data [rs0, rs1], [gs0, gs1].
    Copy from DSUN, CVPR2020, Liu
    """
    def __init__(
        self,
        root_dir,
        seq_len=3,
        load_middle_gs=False,  # ! for gs
        load_flow=False,
        load_optiflow=False,
        load_mask=False,
        load_depth=False,
        img_ext=".png",
        data_aug=ToTensor(),
        training=True,
    ):
        self.load_flow = load_flow
        self.load_mask = load_mask
        self.load_depth = load_depth
        self.load_optiflow = load_optiflow

        self.depth, self.vel = [], []
        self.I_gs, self.I_gs_f, self.I_rs = [], [], []
        self.flow, self.optiflow = [], []
        self.mask = []
        self.seq_len = seq_len

        self.data_aug = copy.deepcopy(data_aug)
        self.training = training

        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != "gt_vel.log":
                    continue

                # read in ground truth velocity
                path_vel = osp.join(seq_path, "gt_vel.log")
                _vel = open(path_vel, "r").readlines()
                for v in _vel:
                    if "#" in v:
                        continue
                    v = v.replace("\n", "")
                    v = v.split(" ")
                    vel = [float(x) * (-1.0) for x in v]

                # read in seq of images
                seq_Irs, seq_Igs, seq_Igs_f, seq_Drs = [], [], [], []
                seq_flow, seq_optiflow = [], []
                seq_mask = []

                for i in range(10):
                    if not osp.isfile(osp.join(seq_path, str(i).zfill(4) + "_rs" + img_ext)):
                        continue

                    seq_Irs.append(osp.join(seq_path, str(i).zfill(4) + "_rs" + img_ext))
                    if load_middle_gs:
                        seq_Igs.append(osp.join(seq_path, str(i).zfill(4) + "_gs_m" + img_ext))
                    else:
                        seq_Igs.append(osp.join(seq_path, str(i).zfill(4) + "_gs_f" + img_ext))

                    seq_Drs.append(osp.join(seq_path, str(i).zfill(4) + "_rs.pdepth"))
                    seq_flow.append(osp.join(seq_path, str(i).zfill(4) + "_flow_raft_m.flo"))
                    seq_mask.append(osp.join(seq_path, str(i).zfill(4) + "_mask" + img_ext))

                    if i == 9:
                        seq_optiflow.append(osp.join(seq_path, str(8).zfill(4) + "_flow_raft.flo"))
                    else:
                        seq_optiflow.append(osp.join(seq_path, str(i).zfill(4) + "_flow_raft.flo"))
                    # if osp.exists(seq_optiflow[0]) or not osp.exists(seq_optiflow[1]):
                    #    seq_optiflow[1] = seq_optiflow[0]

                    if not osp.exists(seq_Irs[-1]):
                        break

                    if len(seq_Irs) < seq_len:
                        continue

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())
                    self.depth.append(seq_Drs.copy())
                    self.vel.append(vel)
                    self.flow.append(seq_flow.copy())
                    self.optiflow.append(seq_optiflow.copy())
                    self.mask.append(seq_mask.copy())

                    seq_Irs.pop(0)
                    seq_Igs.pop(0)
                    seq_Drs.pop(0)
                    seq_flow.pop(0)
                    seq_optiflow.pop(0)
                    seq_mask.pop(0)

    def __len__(self):
        return len(self.vel)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]
        path_mask = self.mask[idx] if self.load_mask else []

        # ! Read images and Data augmentation from seq
        I_rs = list()
        for i in range(self.seq_len):
            I_rs.append(Image.open(path_rs[i]).convert("RGB"))

        I_gs = Image.open(path_gs[self.seq_len // 2]).convert("RGB")
        mask = Image.open(path_mask[self.seq_len // 2]).convert("RGB") if self.load_mask else []

        paths = {
            "RS": str(path_rs[self.seq_len // 2]),
            "GS": str(path_gs[self.seq_len // 2]),
            "MASK": str(path_mask[self.seq_len // 2]),
        }

        I_rs, I_gs, mask = self.data_aug(I_gs.size, I_rs=I_rs, I_gs=I_gs, mask=mask)
        return I_rs, I_gs, mask, paths


def get_data_loaders(
    train_dir="/mnt/lustre/qudelin/DATA/RS/Carla/train",
    val_dir="/mnt/lustre/qudelin/DATA/RS/Carla/val",
    test_dir="/mnt/lustre/qudelin/DATA/RS/Carla/test",
    batch_size=2,
    shuffle=True,
    num_workers=2,
    seq_len=5,
    load_mask=False,
    training=True,
    load_middle_gs=False,  # ! for gs
    data_aug=ToTensor(),
):
    loader_args = {
        "batch_size": batch_size,
        # "shuffle": shuffle,
        "num_workers": num_workers,
    }

    if training:
        train_dataset = Carla(
            train_dir,
            seq_len=seq_len,
            load_mask=load_mask,
            data_aug=data_aug,
            training=training,
            load_middle_gs=load_middle_gs,
        )
        valid_dataset = Carla(
            val_dir,
            seq_len=seq_len,
            load_mask=load_mask,
            data_aug=data_aug,
            training=training,
            load_middle_gs=load_middle_gs,
        )

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
            valid_sampler = DistributedSampler(valid_dataset, shuffle=shuffle)
            print("============== Use DistributedSampler =============")

        return DataLoader(
            train_dataset, shuffle=(train_sampler is None and shuffle), sampler=train_sampler, **loader_args
        ), DataLoader(valid_dataset, shuffle=(valid_sampler is None and shuffle), sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(
            Carla(
                test_dir,
                seq_len=seq_len,
                load_mask=load_mask,
                data_aug=data_aug,
                training=training,
                load_middle_gs=load_middle_gs,
            ),
            **loader_args
        )


if __name__ == "__main__":
    dataloader, *_ = get_data_loaders(load_mask=True, training=True)

    images, gt, mask = next(iter(dataloader))
    print(len(images), images[0].shape, gt.shape, mask.shape)
    print(mask)

    # from torchvision.utils import save_image

    # for i, rs in enumerate(images):
    #     save_image(rs, "rs_{:04d}.png".format(i))
    # save_image(gt, "gs.png")

    # for batch_idx, (data, target, *arg) in enumerate(dataloader):
    #     # print(batch_idx, data[0].shape, target.shape, arg)
    #     print(batch_idx, data[0].shape, target.shape)
