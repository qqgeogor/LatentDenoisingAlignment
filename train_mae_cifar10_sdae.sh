python train_mae_cifar10_sdae.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_sdae/  \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train'  \
    --save_freq 5 \
    --num_views 16 \
    --batch_size 128 \
    --use_checkpoint \
    --epochs 800