

python train_patch_mmcr.py \
    --decoder_embed_dim 192 \
    --use_amp \
    --output_dir /mnt/d/repo/output/output_pmmcr/ \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path /mnt/d/datasets/tiny-imagenet-200 \
    --save_freq 10 \
    --epochs 100 \
    --resume /mnt/d/repo/output/output_pmmcr/checkpoint_20.pth


python train_patch_mmcr.py --use_amp --output_dir ../../autodl-tmp/output_pmmcr/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train' --save_freq 20 --epochs 100

