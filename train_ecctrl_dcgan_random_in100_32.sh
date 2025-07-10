python train_ecctrl_dcgan_random.py     \
    --dataset imagenet100 \
    --data_path ../../autodl-fs/imagenet100     \
    --output_dir ../../autodl-tmp/output_ecctrl_dcgan_random_in100_32     \
    --img_size 32 \
    --save_freq 10     \
    --batch_size 128    \
    --latent_dim 128 \
    --resume ../../autodl-tmp/output_ecctrl_dcgan_random/ebm_gan_checkpoint_800.pth