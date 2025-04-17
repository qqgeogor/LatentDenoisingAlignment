python train_ijepa.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_ijepa_imagenet100/  \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/imagenet100' \
    --mask_ratio 0.75 \
    --use_checkpoint \
    --save_freq 20 \
    --use_amp

