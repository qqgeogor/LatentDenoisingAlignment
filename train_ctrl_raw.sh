python train_ctrl_raw.py     \
    --data_path /mnt/d/datasets/cifar10     \
    --output_dir /mnt/d/repo/output/r3gan-ctrl-raw     \
    --save_freq 10 \
    --latent_dim 384 \
    --adv_weight 1 \
    --use_amp \
    --resume /mnt/d/repo/output/r3gan-ctrl-inv/ebm_gan_checkpoint_20.pth