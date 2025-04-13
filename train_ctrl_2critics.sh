python train_ctrl.py     \
    --data_path /mnt/d/datasets/cifar10     \
    --output_dir /mnt/d/repo/output/r3gan-ctrl-2critics     \
    --save_freq 20 \
    --n_critic 2 \
    --g_lr 1.5e-4 \
    --d_lr 1.5e-4 \
    --batch_size 1024 \
    --use_amp


