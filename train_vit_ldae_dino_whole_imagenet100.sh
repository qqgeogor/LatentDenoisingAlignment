# python train_vit_ldae_dino_whole.py \
#     --use_amp \
#     --output_dir ../../autodl-tmp/output_vit_ldae_dino_whole/  \
#     --img_size 128 \
#     --patch_size 16 \
#     --decoder_embed_dim 96 \
#     --dataset 'tiny-imagenet' \
#     --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
#     --mask_ratio 0.75 \
#     --noise_scale 0.5 \
#     --use_checkpoint \
#     --save_freq 20 


python train_vit_ldae_dino_whole.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_vit_ldae_dino_whole_imagenet100/  \
    --img_size 224 \
    --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/imagenet100/' \
    --embed_dim 384 \
    --decoder_embed_dim 192 \
    --num_heads 6 \
    --decoder_num_heads 3 \
    --mlp_ratio 4 \
    --depth 12 \
    --decoder_depth 4 \
    --mask_ratio 0.75 \
    --noise_scale 0.5 \
    --use_checkpoint \
    --resume ../../autodl-tmp/output_vit_ldae_dino_whole_imagenet100/checkpoint_epoch_20.pth \
    --save_freq 20 


