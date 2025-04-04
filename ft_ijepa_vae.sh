

python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_ijepa_vae/checkpoint_epoch_100.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name ijepa_vae     \
    --run_name ijepa_vae_100epoch

