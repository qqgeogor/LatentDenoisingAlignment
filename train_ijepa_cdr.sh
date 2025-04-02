python train_ijepa_cdr.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_ijepa_cdr/  \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --mask_ratio 0.75 \
    --use_checkpoint \
    --save_freq 20 

