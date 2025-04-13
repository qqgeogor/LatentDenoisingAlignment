python train_ctrl_ldr_ema_cos_cl_temp01.py     \
    --data_path /mnt/d/datasets/cifar10     \
    --output_dir /mnt/d/repo/output/ctrl-ldr-ema-cos-cl-temp01-pca-noise     \
    --save_freq 10    \
    --batch_size 128    \
    --latent_dim 384 \
    --adv_weight 0.2 \
    --resume /mnt/d/repo/output/ctrl-ldr-ema-cos-cl-temp01/ebm_gan_checkpoint_300.pth \
    --noise_scale 0.5
    