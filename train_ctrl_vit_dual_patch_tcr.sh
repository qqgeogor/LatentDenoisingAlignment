

python train_ctrl_vit_dual_patch_tcr.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/  \
    --save_freq 20 \
    --gp_weight 0.05 \
    --adv_weight 0.5 \
    --tcr_weight 0.0 \
    --resume ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/ebm_gan_checkpoint_20.pth \
    --use_amp 