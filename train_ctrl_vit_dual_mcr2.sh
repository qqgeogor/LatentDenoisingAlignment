
python train_ctrl_vit_dual_mcr2.py     \
    --data_path ../../autodl-tmp/cifar10     \
    --output_dir ../../autodl-tmp/output_ctrl_vit_dual_mcr2/  \
    --save_freq 20 \
    --gp_weight 0.5 \
    --batch_size 1024 \
    --use_checkpoint