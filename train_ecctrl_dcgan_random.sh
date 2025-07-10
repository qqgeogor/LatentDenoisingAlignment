python train_ecctrl_dcgan_random.py     \
    --dataset cifar10 \
    --data_path ../../autodl-fs/cifar10     \
    --output_dir ../../autodl-tmp/output_ecctrl_dcgan_random     \
    --img_size 32 \
    --save_freq 10     \
    --batch_size 128    \
    --latent_dim 128 \
    --resume ../../autodl-tmp/output_ecctrl_dcgan_random/ebm_gan_checkpoint_800.pth