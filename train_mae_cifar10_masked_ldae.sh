python train_mae_cifar10_masked_ldae.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_masked_ldae/  \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.5 \
    --save_freq 20 

