python train_mae_cifar10_ldae.py \
--use_amp \
--output_dir ../../autodl-fs/output_ldae/  \
--img_size 128 \
--patch_size 16 \
--dataset 'tiny-imagenet' \
--data_path '../../autodl-tmp/tiny-imagenet-200/' \
--noise_scale 1.5 \
--save_freq 100