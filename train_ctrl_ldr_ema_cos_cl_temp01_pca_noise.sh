python train_ctrl_ldr_ema_cos_cl_temp01.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_temp01_pca_noise     \
    --save_freq 10    \
    --batch_size 128    \
    --latent_dim 384 \
    --adv_weight 0.2 \
    --noise_scale 0.5 \
    --resume ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_temp01/ebm_gan_checkpoint_400.pth
    