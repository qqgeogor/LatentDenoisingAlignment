

python train_ctrl_vit_dual_patch_tcr.py     \
    --dataset imagenet100 \
    --data_path ../../autodl-tmp/imagenet100     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/  \
    --img_size 128 \
    --patch_size 16 \
    --embed_dim 192 \
    --depth 12 \
    --num_heads 3 \
    --decoder_embed_dim 192 \
    --decoder_depth 4 \
    --decoder_num_heads 3 \
    --save_freq 10 \
    --gp_weight 0.05 \
    --adv_weight 0.2 \
    --tcr_weight 0.0 \
    --global_weight 0.0 \
    --use_amp 