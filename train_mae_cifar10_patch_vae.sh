


# python train_mae_cifar10_patch_vae.py \
#     --decoder_embed_dim 192 \
#     --use_amp \
#     --output_dir /mnt/d/repo/output/output_patch_vdae/ \
#     --pretrained_vae /mnt/d/repo/output/output_pvae/checkpoint_epoch_30.pth \
#     --img_size 128 \
#     --patch_size 16 \
#     --dataset 'tiny-imagenet' \
#     --data_path /mnt/d/datasets/tiny-imagenet-200 \
#     --save_freq 1



python train_mae_cifar10_patch_vae.py --use_amp --output_dir ../../autodl-tmp/output_patch_vdae/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train' --save_freq 100
