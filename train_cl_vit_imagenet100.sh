python train_cl_vit.py     \
    --dataset imagenet100 \
    --img_size 128 \
    --patch_size 16 \
    --embed_dim 192 \
    --depth 12 \
    --num_heads 3 \
    --decoder_embed_dim 192 \
    --decoder_depth 4 \
    --decoder_num_heads 3 \
    --mlp_ratio 4 \
    --data_path ../../autodl-tmp/imagenet100     \
    --output_dir ../../autodl-tmp/output_cl_vit_imagenet100  \
    --save_freq 10    \
    --batch_size 128    \
    --latent_dim 192 \
    --adv_weight 0.2 \
    --use_amp