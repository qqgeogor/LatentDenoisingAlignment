python train_ctrl_ldr_ema_cos_cl.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_2critics     \
    --save_freq 20     \
    --batch_size 128    \
    --n_critic 2     \
    --latent_dim 384 \
    --resume ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl/ebm_gan_checkpoint_80.pth