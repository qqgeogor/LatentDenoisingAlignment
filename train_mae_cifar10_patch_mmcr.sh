


python train_mae_cifar10_patch_mmcr.py \
    --decoder_embed_dim 192 \
    --use_amp \
    --output_dir /mnt/d/repo/output/output_patch_vdae/ \
    --pretrained_mmcr /mnt/d/repo/output/output_pmmcr/checkpoint_epoch_40.pth \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path /mnt/d/datasets/tiny-imagenet-200 \
    --save_freq 1


python train_mae_cifar10_patch_mmcr.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_patch_vdae/  \
    --img_size 128 \
    --patch_size 16 \
    --pretrained_mmcr ../../autodl-tmp/output_pmmcr/checkpoint_epoch_99.pth \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --save_freq 20
    