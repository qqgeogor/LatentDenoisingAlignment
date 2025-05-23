python train_vit_ldae_dino_whole.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_vit_ldae_dino_whole/  \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 96 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --noise_scale 0.5 \
    --use_checkpoint \
    --save_freq 20 \
    --resume ../../autodl-tmp/output_vit_ldae_dino_whole/checkpoint_epoch_160.pth

