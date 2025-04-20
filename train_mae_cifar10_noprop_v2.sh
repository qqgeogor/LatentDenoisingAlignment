python train_mae_cifar10_noprop.py \
    --use_amp \
    --output_dir ../../autodl-fs/output_mae_noprop2/  \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --T 10 \
    --decoder_depth 2 \
    --save_freq 20 

    