python train_mae_cifar10_mae_cnn.py \
    --use_amp \
    --output_dir ../../autodl-tmp/output_mae_cnn_repvgg/  \
    --img_size 32 \
    --patch_size 4 \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type repvgg \
    --dataset 'cifar10' \
    --data_path '../../autodl-fs/cifar10' \
    --mask_ratio 0.75 \
    --num_views 1 \
    --save_freq 20 
