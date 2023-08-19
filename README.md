# Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction (under construction)

### [Paper](https://arxiv.org/pdf/2303.18125.pdf) | [Project Page](https://delinqu.github.io/QRSC) | [Video](https://youtu.be/Or-yvKHUrZ0)

> Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction <br />
> [Delin Qu*](https://delinqu.github.io/), Yizhen Lao*, Zhigang Wang*, Dong Wang, Bin Zhao†, Xuelong Li†,
> ICCV 2023

<!-- <p align="center">
  <a href="">
    <img src="./media/coslam_teaser.gif" alt="Logo" width="80%">
  </a>
</p> -->


This repository contains the code for the paper Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction, a geometry-based quadratic rolling shutter motion solver that precisely estimates the high-order correction field of individual pixels and a self-alignment 3D video architecture $RSA^2$-Net for high-quality frame aggregation and synthesis against extreme scene occlusion.

## Update
- [x] Code for QRSC [2023-8-03]
- [x] Project Pages for Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction [2023-8-16]
- [x] Youtube video for Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction [2023-8-16]

## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/DelinQu/QRSC.git
cd QRSC
pip install -r requirements.txt
```

### Install the environment

```bash
# Create conda environment
conda create -n qrsc python=3.9.12
conda activate qrsc

# Install all the dependencies via pip
pip install -r requirements.txt

# Build extensions: LPIPS and package_core for evaluation
cd src/lib
bash install.sh
```

### Install MMFlow
Please refer to the [installation](https://github.com/open-mmlab/mmflow/blob/master/docs/en/install.md) for optical flow support. After that, please download the RAFT and GMA model by:
```bash
mim download mmflow --config gma_plus-p_8x2_120k_mixed_368x768
mim download mmflow --config raft_8x2_100k_mixed_368x768
```

The model will be automatically downloaded at $HOME/.cache

## Dataset
Plase follow [DeepUnrollNet](https://github.com/ethliup/DeepUnrollNet) and [BSRSC](https://github.com/ljzycmd/BSRSC) to download the Carla-RS, Fastec-RS and BSRSC datasets. The update the dataset [configurations](conf/dataset):
```yaml
# @package data_loader
_target_: src.dataset.Carla.get_data_loaders
train_dir: path_to_carla_train
val_dir: path_to_carla_val
test_dir: path_to_carla_test
batch_size: ${batch_size}
seq_len: ${arch.n_inputs}
load_mask: true
training: true
shuffle: true
num_workers: ${n_cpu}
load_middle_gs: true

data_aug: 
  _target_: src.dataset.transforms.ToTensor
```

## Evaluation
The [script](run_test.sh) provides evaluation methods of all the three datasets. Before run the following code, please [download the checkpoints]() to the [checkpoint dir](checkpoint).

```bash
bash run_test.sh
```

## Traning
We provides the training [scripts](run.sh) on all the three datasets.
```bash
bash run_test.sh
```

## Acknowledgement

We adapt codes from some awesome repositories, including [BS-RSC](https://github.com/ljzycmd/BSRSC), [Video-Frame-Interpolation-Transformer](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer) and [DeepUnrollNet](https://github.com/ethliup/DeepUnrollNet). Thanks for making the code available.

The research presented here has been supported by ...


## Citation

If you find our code or paper useful for your research, please consider citing:

```
@inproceedings{qu2023towards,
        title={Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction},
        author={Delin Qu, Yizhen Lao, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li},
        booktitle={ICCV},
        year={2023}
}
```
