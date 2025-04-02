python train_vit_ldae_dino_self_dist_l1.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_tiny_self_dist_l1/  \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --use_checkpoint \
    --save_freq 20 
