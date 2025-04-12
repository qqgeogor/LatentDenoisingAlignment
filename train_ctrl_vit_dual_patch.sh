
python train_ctrl_vit_dual_patch.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_patch_pca/  \
    --save_freq 10 \
    --gp_weight 0.5 \
    --noise_scale 0 \
    --use_amp \
    --resume ../../autodl-tmp/output_ctrl_vit_dual_patch/ebm_gan_checkpoint_30.pth



python train_ctrl_vit_dual_patch.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_patch/  \
    --save_freq 10 \
    --gp_weight 0.5 \
    --noise_scale 0.5 \
    --use_amp \
    --resume ../../autodl-tmp/output_ctrl_vit_dual_patch/ebm_gan_checkpoint_30.pth