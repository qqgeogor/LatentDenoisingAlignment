python train_mae_cifar10_jepa_l1.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_jepa_l1/  \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --num_views 1 \
    --save_freq 20 