python train_mae_cifar10_jepa_vae.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_jepa_vae/  \
    --pretrained_vae ../../autodl-tmp/output_pvae/checkpoint_epoch_99.pth \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --num_views 1 \
    --save_freq 20 
    