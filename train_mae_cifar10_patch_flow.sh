


# python train_mae_cifar10_patch_flow.py \
#     --decoder_embed_dim 192 \
#     --use_amp \
#     --output_dir /mnt/d/repo/output/output_patch_vdae/ \
#     --pretrained_flow /mnt/d/repo/output/output_pflow/checkpoint_epoch_30.pth \
#     --img_size 128 \
#     --patch_size 16 \
#     --dataset 'tiny-imagenet' \
#     --data_path /mnt/d/datasets/tiny-imagenet-200 \
#     --save_freq 1


python train_mae_cifar10_patch_flow.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_patch_flow_dae/  \
    --img_size 128 \
    --patch_size 16 \
    --pretrained_flow ../../autodl-tmp/output_pflow/checkpoint_epoch_99.pth \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --save_freq 20
    