python train_mae_cifar10_noprop.py \
    --use_amp \
    --output_dir ../../autodl-fs/output_mae_noprop3/  \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --T 1 \
    --decoder_depth 4 \
    --save_freq 20 