python train_mae_cifar10_hldae_imagenet_sigma_0_333_sqrt_noise_pred.py \
--use_amp \
--output_dir ../../autodl-tmp/output_ae_sigma_0_25_sqrt_noise_pred/  \
--img_size 128 \
--patch_size 16 \
--dataset 'tiny-imagenet' \
--data_path '../../autodl-tmp/tiny-imagenet-200/train' \
--noise_scale 0.5 \
--save_freq 20 