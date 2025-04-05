python train_mae_cifar10_hldae_imagenet_sigma_0_5_sqrt_noise_pred_uniform.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_ae_sigma_0_5_sqrt_noise_pred_uniform/ \
    --img_size 128 --patch_size 16 \
    --dataset 'tiny-imagenet' \
    --data_path '../../autodl-tmp/tiny-imagenet-200/train' \
    --save_freq 20 \
    --noise_scale 0.5
