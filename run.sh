# * === Carla ===
python train.py \
    hparams=lr2e4_step25 \
    dataset=carla_train \
    model=qrst_gma \
    arch.n_inputs=5 \
    loss=lc_lp_mse_auto \
    optimizer=adam \
    status=train \
    gamma=0.9999999744 \
    delta=0.9999999744 \
    tau=0.4999999872 \
    name=Carla_qrst_gma_lr2e4_step25


# * === Fastec ===
python train.py \
    hparams=lr2e4_step25 \
    dataset=fastec_train \
    model=qrst_raft \
    arch.n_inputs=5 \
    loss=lc_lp_mse_auto \
    optimizer=adam \
    status=train \
    gamma=1.0 \
    delta=1.0 \
    tau=0.5 \
    name=Fastec_qrst_raft_lr2e4_step25


# * === BSRSC ===
python train.py \
    hparams=lr1e4_step25 \
    dataset=bsrsc_train \
    model=qrst_raft \
    arch.n_inputs=5 \
    loss=lc_lp_mse_auto \
    optimizer=adam \
    status=test \
    gamma=0.45 \
    delta=1.0 \
    tau=0.225 \
    name=BSRSC_qrst_raft_lr1e4_step25
