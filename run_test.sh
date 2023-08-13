# * === Carla ===
python evaluate.py \
    hparams=lr2e4_step25 \
    dataset=carla_test \
    model=qrst_gma \
    arch.n_inputs=5 \
    status=test \
    gamma=0.9999999744 \
    delta=0.9999999744 \
    tau=0.4999999872 \
    checkpoint=/mnt/petrelfs/qudelin/PJLAB/QRST/QRSC/checkpoint/carla.pth


# * === Fastec ===
python evaluate.py \
    hparams=lr2e4_step25 \
    dataset=fastec_test \
    model=qrst_raft \
    arch.n_inputs=5 \
    status=test \
    gamma=1.0 \
    delta=1.0 \
    tau=0.5 \
    checkpoint=/mnt/petrelfs/qudelin/PJLAB/QRST/QRSC/checkpoint/fasctec.pth


# * === BSRSC ===
python evaluate.py \
    hparams=lr1e4_step25 \
    dataset=bsrsc_test \
    model=qrst_raft \
    arch.n_inputs=5 \
    status=test \
    gamma=0.45 \
    delta=1.0 \
    tau=0.225 \
    checkpoint=/mnt/petrelfs/qudelin/PJLAB/QRST/QRSC/checkpoint/bsrsc.pth
