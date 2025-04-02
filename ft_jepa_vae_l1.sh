

python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_jepa_vae_l1/checkpoint_epoch_400.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name jepa_vae_l1     \
    --run_name jepa_vae_l1_400epoch

