



python train_patch_vae.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_pvae_imagenet100/  \
    --img_size 224 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/imagenet100/' \
    --save_freq 20 \
    --embed_dim 384 \
    --epochs 100

