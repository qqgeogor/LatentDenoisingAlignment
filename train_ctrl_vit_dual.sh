
python train_ctrl_vit_dual.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual/  \
    --save_freq 10 \
    --gp_weight 0.5 \
    --use_amp \
    --resume ../../autodl-tmp/output_ctrl_vit_dual/ebm_gan_checkpoint_20.pth