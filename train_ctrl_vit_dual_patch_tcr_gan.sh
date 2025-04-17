
python train_ctrl_vit_dual_patch_tcr_gan.py     \
    --dataset imagenet100 \
    --data_path /mnt/d/datasets/imagenet100/train       \
    --output_dir /mnt/d/repo/output/output_ctrl_vit_dual_patch_tcr_gan/  \
    --img_size 128 \
    --patch_size 16 \
    --embed_dim 192 \
    --depth 12 \
    --num_heads 3 \
    --decoder_embed_dim 192 \
    --decoder_depth 4 \
    --decoder_num_heads 3 \
    --save_freq 10 \
    --d_adv_weight 0.2 \
    --g_adv_weight 0.2 \
    --gp_weight 0.05 \
    --tcr_weight 1.0 \
    --global_weight 0.0 \
    --temperature 1 \
    --num_workers 8 \
    --use_amp 
    