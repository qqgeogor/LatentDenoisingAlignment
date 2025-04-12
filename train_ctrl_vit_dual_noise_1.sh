
python train_ctrl_vit_dual.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_noise_1/  \
    --save_freq 10 \
    --gp_weight 0.05 \
    --noise_scale 1 \
    --use_amp