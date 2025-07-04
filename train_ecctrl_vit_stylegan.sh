python train_ecctrl_vit_stylegan.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ecctrl_vit_stylegan     \
    --save_freq 10     \
    --batch_size 128    \
    --latent_dim 192 \
    --embed_dim 192 \
    --depth 12 \
    --num_heads 3 \
    --decoder_depth 4 \
    --decoder_embed_dim 96 \
    --tcr_weight 0.01 \
    --img_size 32 \
    --patch_size 4 