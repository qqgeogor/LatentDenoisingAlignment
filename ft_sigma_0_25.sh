
python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_ae_sigma_0_25_sqrt_noise_pred/checkpoint_epoch_200.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name ae_sigma_0_25_sqrt_noise_pred     \
    --run_name ae_sigma_0_25_sqrt_noise_pred_200epoch
