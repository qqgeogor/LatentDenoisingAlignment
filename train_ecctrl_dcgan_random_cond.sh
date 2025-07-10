python train_ecctrl_dcgan_random_cond.py     \
    --dataset cifar10 \
    --data_path ../../autodl-fs/cifar10     \
    --output_dir ../../autodl-tmp/output_ecctrl_dcgan_random_cond     \
    --img_size 32 \
    --save_freq 10     \
    --batch_size 128    \
    --latent_dim 128 \
    --use_condition
