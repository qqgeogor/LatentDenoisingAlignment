python train_ecctrl_dcgan_random_clip.py     \
    --dataset imagenet100 \
    --data_path ../../autodl-fs/imagenet100     \
    --output_dir ../../autodl-tmp/output_ecctrl_dcgan_random_clip_in100_32     \
    --img_size 32 \
    --save_freq 100     \
    --batch_size 128    \
    --latent_dim 768 \
    --clip_model ViT-B/32