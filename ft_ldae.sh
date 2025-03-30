python finetune_cifar_classifier_hldae_dino.py --pretrained_path ../../autodl-tmp/output/*_40.pth 



# python finetune_cifar_classifier_hldae_dino.py --pretrained_path /mnt/d/repo/vit_ldae_tiny/checkpoint_epoch_10.pth --img_size 128 --patch_size 16


python finetune_cifar_classifier_hldae_dino.py --pretrained_path ../../autodl-tmp/output_tiny/*_10.pth  --img_size 128 --patch_size 16




python finetune_cifar_classifier_vit_hldae_tiny.py --pretrained_path /mnt/d/repo/output/cifar_self_dist/checkpoint_epoch_1.pth  --img_size 128 --patch_size 16 --decoder_embed_dim 192


python finetune_cifar_classifier_vit_hldae_tiny.py \
 --pretrained_path ../../autodl-tmp/output_mae_imagenet100/checkpoint_epoch_100.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name mae_imagenet \
 --run_name mae_imagenet100_100epoch


 python finetune_cifar_classifier_vit_hldae_tiny.py \
 --pretrained_path ../../autodl-tmp/output_edm_imagenet100/checkpoint_epoch_100.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name edm_imagenet \
 --run_name edm_imagenet100_100epoch



 
python finetune_cifar_classifier_vit_hldae_tiny.py \
 --pretrained_path ../../autodl-tmp/output_mae_imagenet100/checkpoint_epoch_200.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name mae_imagenet \
 --run_name mae_imagenet100_200epoch


 python finetune_cifar_classifier_vit_hldae_tiny.py \
 --pretrained_path ../../autodl-tmp/output_edm_imagenet100/checkpoint_epoch_200.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name edm_imagenet \
 --run_name edm_imagenet100_200epoch





python finetune_cifar_classifier_vit_hldae_tiny.py \
    --pretrained_path ../../autodl-tmp/output_patch_vdae/checkpoint_epoch_1000.pth \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 96 \
    --decoder_num_heads 3 \
    --mlflow_experiment_name patch_vdae \
    --run_name patch_vdae_1000epoch




python finetune_cifar_classifier_vit_hldae_tiny.py \
    --pretrained_path ../../autodl-tmp/output_mae/checkpoint_epoch_1000.pth \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 96 \
    --decoder_num_heads 3 \
    --mlflow_experiment_name mae \
    --run_name mae_1000epoch




python finetune_cifar_classifier_vit_hldae_tiny.py \
 --pretrained_path ../../autodl-tmp/output_mae_imagenet100/checkpoint_epoch_400.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name mae_imagenet \
 --run_name mae_imagenet100_400epoch


 python finetune_cifar_classifier_vit_edm.py \
 --pretrained_path ../../autodl-tmp/output_edm_imagenet100/checkpoint_epoch_400.pth \
 --img_size 224 \
 --patch_size 16 \
 --embed_dim 384 \
 --num_heads 6 \
 --decoder_embed_dim 192 \
 --decoder_num_heads 3 \
 --mlflow_experiment_name edm_imagenet \
 --run_name edm_imagenet100_400epoch