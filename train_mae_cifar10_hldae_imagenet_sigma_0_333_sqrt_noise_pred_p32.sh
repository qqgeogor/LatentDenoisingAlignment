python train_mae_cifar10_hldae_imagenet_sigma_0_333_sqrt_noise_pred.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_pca_patch_32/  \
    --img_size 128 \
    --patch_size 16 \
    --noiser_patch_size 32 \
    --decoder_embed_dim 96 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --noise_scale 0.5 \
    --use_checkpoint \
    --save_freq 20 