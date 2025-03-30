

python train_patch_vae.py \
    --decoder_embed_dim 192 \
    --use_amp \
    --output_dir /mnt/d/repo/output/output_pvae/ \
    --img_size 128 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path /mnt/d/datasets/tiny-imagenet-200 \
    --save_freq 10 \
    --resume /mnt/d/repo/output/output_pvae/checkpoint_30.pth \
    --epochs 100


python train_patch_vae.py --use_amp --output_dir ../../autodl-tmp/output_pvae/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train' --save_freq 20 --epochs 100

