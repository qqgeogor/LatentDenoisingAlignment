python train_cl_dino.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_cl_dino     \
    --save_freq 20     \
    --batch_size 128    \
    --epochs 301 \
    --latent_dim 384 \
    --resume ../../autodl-tmp/output_cl_dino/ebm_gan_checkpoint_80.pth