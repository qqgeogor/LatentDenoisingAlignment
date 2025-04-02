

python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_tiny_self_dist_l1/checkpoint_epoch_100.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name hldae_self_dist_l1     \
    --run_name hldae_self_dist_l1_100epoch

