
python finetune_ctrl.py     \
    --pretrained_path /mnt/d/repo/output/cifar10-ebm-gan-r3gan-ctrl/ebm_gan_checkpoint_160.pth     \
    --img_size 32 \
    --use_amp 



# python finetune_ctrl_vit.py     \
#     --pretrained_path /mnt/d/repo/output/r3gan-ctrl-vit/ebm_gan_checkpoint_140.pth     \
#     --img_size 32 \
#     --use_amp \
#     --freeze_backbone 



python finetune_ctrl.py     \
    --pretrained_path ../../autodl-tmp/output_ctrl_mcr2/ebm_gan_checkpoint_100.pth     \
    --img_size 32 \
    --use_amp 




python finetune_ctrl.py     \
    --img_size 32 \
    --latent_dim 384 \
    --use_amp \
    --pretrained_path /mnt/d/repo/output/r3gan-ctrl-ldr-r3gan/ebm_gan_checkpoint_100.pth     \
    

python finetune_ctrl.py     \
    --img_size 32 \
    --latent_dim 384 \
    --use_amp \
    --pretrained_path /mnt/d/repo/output/r3gan-ctrl-ldr-r3gan/ebm_gan_checkpoint_200.pth     \
    


python finetune_ctrl_ldr_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path /mnt/d/repo/output/r3gan-ctrl-ldr-r3gan-dino/ebm_gan_checkpoint_250.pt



python finetune_ctrl_knn.py \
    --use_amp     \
    --pretrained_path /mnt/d/repo/output/r3gan-ctrl/ebm_gan_checkpoint_1000.pth






python finetune_ctrl_knn.py \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_mcr2/ebm_gan_checkpoint_1000.pth 





python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae/ebm_gan_checkpoint_700.pth \
    --visualize_features


python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae/ebm_gan_checkpoint_760.pth \
    --visualize_features

python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae_5critics/ebm_gan_checkpoint_760.pth \
    --visualize_features

python finetune_ctrl_ldr_vae.py \
    --latent_dim 384     \
    --use_amp 



python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ae_ldr/ebm_gan_checkpoint_20.pth \
    --visualize_features




python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ae_ldr_patch_pca/ebm_gan_checkpoint_20.pth \
    --visualize_features



    

python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae_patch_pca/ebm_gan_checkpoint_760.pth \
    --visualize_features





python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_cl_dino/ebm_gan_checkpoint_20.pth \
    --visualize_features



python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae_patch_pca_aug_v3/ebm_gan_checkpoint_20.pth \
    --visualize_features
    

python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae_patch_pca_aug/ebm_gan_checkpoint_280.pth \
    --visualize_features
    

python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_r3gan_vae_patch_pca_aug_v6/ebm_gan_checkpoint_20.pth \
    --visualize_features
    




python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_vit_dual/ebm_gan_checkpoint_240.pth \
    --visualize_features
    


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_vit/ebm_gan_checkpoint_10.pth \
    --visualize_features
    



python finetune_ctrl_ldr_vae_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path /mnt/d/repo/output/ctrl-ldr-ema-cos-cl-temp01/ebm_gan_checkpoint_10.pth \
    --visualize_features
    



python finetune_ctrl_ldr_vae.py     \
    --latent_dim 384         \
    --use_amp   \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_temp01/ebm_gan_checkpoint_300.pth \
    --mlflow_experiment_name ctrl_ldr_ema_cos_cl_temp01 \
    --run_name ctrl_ldr_ema_cos_cl_temp01_300epoch



python finetune_ctrl_ldr_vae.py     \
    --latent_dim 384         \
    --use_amp   \
    --pretrained_path ../../autodl-tmp/output_cl_dino_all/ebm_gan_checkpoint_300.pth \
    --mlflow_experiment_name cl_dino_all \
    --run_name cl_dino_all_300epoch



python finetune_ctrl_ldr_vae.py     \
    --latent_dim 384         \
    --use_amp   \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_temp05/ebm_gan_checkpoint_300.pth \
    --mlflow_experiment_name ctrl_ldr_ema_cos_cl_temp05 \
    --run_name ctrl_ldr_ema_cos_cl_temp05_300epoch


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit/ebm_gan_checkpoint_20.pth   \
    --visualize_features
    


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit/ebm_gan_checkpoint_400.pth   \
    --visualize_features
    



python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_60.pth   \
    --visualize_features 


python finetune_vit_knn.py   \
  --latent_dim 192         \
  --img_size 128     \
  --patch_size 16     \
  --use_amp         \
  --pretrained_path ../../autodl-tmp/output_mdae/checkpoint_epoch_300.pth       \
  --visualize_features  \
  --decoder_depth 4


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_60.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --mlflow_experiment_name ctrl_ldr_ema_cos_cl_vit_imagenet100         \
  --run_name ctrl_ldr_ema_cos_cl_vit_imagenet100_60epoch


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_50.pth   \
    --visualize_features  \
    --num_register_tokens 0


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_reg_imagenet100/ebm_gan_checkpoint_50.pth   \
    --visualize_features  \
    --num_register_tokens 16



python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_90.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name ctrl_ldr_ema_cos_cl_vit_imagenet100         \
  --run_name ctrl_ldr_ema_cos_cl_vit_imagenet100_90epoch


  
python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_reg_imagenet100/ebm_gan_checkpoint_90.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 16 \
  --mlflow_experiment_name ctrl_ldr_ema_cos_cl_vit_reg_imagenet100         \
  --run_name ctrl_ldr_ema_cos_cl_vit_reg_imagenet100_90epoch


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_reg_imagenet100/ebm_gan_checkpoint_90.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 16 \
  --mlflow_experiment_name ctrl_ldr_ema_cos_cl_vit_reg_imagenet100         \
  --run_name ctrl_ldr_ema_cos_cl_vit_reg_imagenet100_90epoch


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_cl_vit_imagenet100/ebm_gan_checkpoint_90.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name cl_vit_imagenet100         \
  --run_name cl_vit_imagenet100_90epoch


  
python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_cl_vit_imagenet100/ebm_gan_checkpoint_90.pth   \
    --visualize_features  \
    --num_register_tokens 0





python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_cl_vit_imagenet100/ebm_gan_checkpoint_90.pth   \
    --visualize_features  \
    --num_register_tokens 0
    


  
python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_90.pth   \
    --visualize_features  \
    --num_register_tokens 0




python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_reg_imagenet100/ebm_gan_checkpoint_90.pth   \
    --visualize_features  \
    --num_register_tokens 16





  
python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_50.pth   \
    --visualize_features  \
    --num_register_tokens 0




python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_cl_vit_imagenet100/ebm_gan_checkpoint_100.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name cl_vit_imagenet100         \
  --run_name cl_vit_imagenet100_100epoch


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_100.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name ctrl_ldr_ema_cos_cl_vit_imagenet100         \
  --run_name ctrl_ldr_ema_cos_cl_vit_imagenet100_100epoch


  
python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_60.pth   \
    --visualize_features  \
    --num_register_tokens 0



  
python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100/ebm_gan_checkpoint_70.pth   \
    --visualize_features  \
    --num_register_tokens 0





python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_cl_vit_ibot_imagenet100/ebm_gan_checkpoint_30.pth   \
    --visualize_features  \
    --num_register_tokens 0


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_cl_vit_ibot_imagenet100/ebm_gan_checkpoint_20.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_cl_vit_ibot_imagenet100         \
  --run_name output_cl_vit_ibot_imagenet100_20epoch


python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_cl_vit_ibot_imagenet100/ebm_gan_checkpoint_90.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_cl_vit_ibot_imagenet100         \
  --run_name output_cl_vit_ibot_imagenet100_90epoch




python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/ebm_gan_checkpoint_10.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --decoder_depth 0 \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_ctrl_vit_dual_patch_tcr         \
  --run_name output_ctrl_vit_dual_patch_tcr_10epoch


  

python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path /mnt/d/repo/output/output_ctrl_vit_dual_patch_tcr/ebm_gan_checkpoint_10.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --decoder_depth 0 \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_ctrl_vit_dual_patch_tcr         \
  --run_name output_ctrl_vit_dual_patch_tcr_10epoch



python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_cl_vit_ibot_imagenet100/ebm_gan_checkpoint_160.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_cl_vit_ibot_imagenet100         \
  --run_name output_cl_vit_ibot_imagenet100_160epoch




python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_cl_vit_ibot_imagenet100/ebm_gan_checkpoint_160.pth   \
    --visualize_features  \
    --num_register_tokens 0


python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/ebm_gan_checkpoint_40.pth   \
    --visualize_features  \
    --num_register_tokens 0



python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr_closeloop/ebm_gan_checkpoint_40.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_ctrl_vit_dual_patch_tcr_closeloop         \
  --run_name output_ctrl_vit_dual_patch_tcr_closeloop_40epoch



python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path ../../autodl-tmp/output_ctrl_vit_dual_patch_tcr/ebm_gan_checkpoint_100.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_ctrl_vit_dual_patch_tcr         \
  --run_name output_ctrl_vit_dual_patch_tcr_100epoch



python finetune_cifar_classifier_vit_ctrl.py       \
  --pretrained_path /mnt/d/repo/output/output_ctrl_vit_dual_patch_tcr_gan/ebm_gan_checkpoint_10.pth         \
  --img_size 128         \
  --patch_size 16          \
  --decoder_embed_dim 192         \
  --decoder_num_heads 3         \
  --decoder_depth 0 \
  --num_register_tokens 0 \
  --mlflow_experiment_name output_ctrl_vit_dual_patch_tcr_gan         \
  --run_name output_ctrl_vit_dual_patch_tcr_gan_10epoch




python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_edm_ijepa_adv_imagenet100/checkpoint_epoch_200.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name edm_ijepa_adv_imagenet100     \
    --run_name edm_ijepa_adv_imagenet100_200epoch


python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_edm_imagenet100/checkpoint_epoch_200.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name edm_ijepa_imagenet100     \
    --run_name edm_ijepa_imagenet100_200epoch




python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path /mnt/d/repo/output/output_edm_ijepa_adv_imagenet100/checkpoint_epoch_200.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name output_edm_ijepa_adv_imagenet100     \
    --run_name output_edm_ijepa_adv_imagenet100_200epoch




python finetune_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_edm_imagenet100/checkpoint_epoch_200.pth   \
    --visualize_features 


python finetune_vit_knn.py \
    --latent_dim 192     \
    --img_size 128 \
    --patch_size 16 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_edm_ijepa_adv_imagenet100/checkpoint_epoch_200.pth   \
    --visualize_features 






python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path /mnt/d/repo/output/output_edm_ijepa_tcr_imagenet100/checkpoint_epoch_40.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 192     \
    --decoder_num_heads 3     \
    --mlflow_experiment_name output_edm_ijepa_tcr_imagenet100     \
    --run_name output_edm_ijepa_tcr_imagenet100_40epoch



python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path ../../autodl-tmp/output_edm_ijepa_adv_vit_small_imagenet100/checkpoint_epoch_100.pth     \
    --img_size 224     \
    --patch_size 16      \
    --embed_dim 384 \
    --decoder_embed_dim 384 \
    --num_heads 6 \
    --decoder_num_heads 6 \
    --mlp_ratio 4 \
    --depth 12 \
    --decoder_depth 4 \
    --mlflow_experiment_name output_edm_ijepa_adv_vit_small_imagenet100     \
    --run_name output_edm_ijepa_adv_vit_small_imagenet100_100epoch





python finetune_cifar_classifier_vit_hldae_tiny.py     \
    --pretrained_path /mnt/d/repo/output/output_vit_ldae_dino_self_dist/checkpoint_epoch_30.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96\
    --decoder_num_heads 3     \
    --mlflow_experiment_name output_vit_ldae_dino_self_dist     \
    --run_name output_vit_ldae_dino_self_dist_30epoch





python finetune_ctrl_vit_knn.py \
    --latent_dim 192     \
    --img_size 32 \
    --patch_size 4 \
    --depth 6 \
    --decoder_embed_dim 192 \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ctrl_ldr_ema_cos_cl_vit_imagenet100_32/ebm_gan_checkpoint_90.pth   \
    --visualize_features 






python finetune_cifar_classifier_vit_noprop.py     \
    --pretrained_path ../../autodl-fs/output_mae_noprop3/checkpoint_epoch_79.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96\
    --decoder_num_heads 3     \
    --mlflow_experiment_name noprop3     \
    --run_name noprop3_79epoch \
    --T 1 \
    --decoder_depth 4




python finetune_cifar_classifier_vit_noprop.py     \
    --pretrained_path ../../autodl-fs/output_mae_noprop/checkpoint_epoch_79.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96\
    --decoder_num_heads 3     \
    --mlflow_experiment_name noprop1     \
    --run_name noprop1_79epoch \
    --T 4 \
    --decoder_depth 1




python finetune_cifar_classifier_vit_noprop.py     \
    --pretrained_path ../../autodl-fs/output_mae_noprop2/checkpoint_epoch_79.pth     \
    --img_size 128     \
    --patch_size 16      \
    --decoder_embed_dim 96\
    --decoder_num_heads 3     \
    --mlflow_experiment_name noprop2     \
    --run_name noprop2_79epoch \
    --T 10 \
    --decoder_depth 2 \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn.py \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp



python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp



python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp



python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp


python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_repvgg/checkpoint_epoch_40.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type repvgg \
    --use_amp



python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --encoder_type sparse_cnn \
    --decoder_embed_dim 192 \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_vit_dec/checkpoint_epoch_180.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp



python finetune_cifar_classifier_vit_mae_cnn.py \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp




python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_in100/checkpoint_epoch_20.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse_in100/checkpoint_epoch_180.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type sparse_cnn \
    --use_amp



python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn/checkpoint_epoch_180.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp




python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_in100/checkpoint_epoch_160.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp



python finetune_cifar_classifier_vit_mae_cnn.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_in100/checkpoint_epoch_1500.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type resnet \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn_v2.py \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp
    
python finetune_cifar_classifier_vit_jepa_cnn_v2.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse_v2/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp




python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse_in100_16p/checkpoint_epoch_20.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type sparse_cnn \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse_in100_16p/checkpoint_epoch_40.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type sparse_cnn \
    --use_amp


python finetune_cifar_classifier_vit_jepa_cnn.py \
    --pretrained_path ../../autodl-tmp/output_jepa_cnn_sparse_in100_16p/checkpoint_epoch_400.pth     \
    --img_size 128     \
    --patch_size 16      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type sparse_cnn \
    --use_amp


python finetune_cifar_classifier_vit_mae_cnn_spconv.py \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type spconv \
    --use_amp





python finetune_cifar_classifier_vit_mae_cnn_spconv.py \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type spconv \
    --use_amp


python finetune_cifar_classifier_vit_mae_cnn_spconv.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_spconv/checkpoint_epoch_1200.pth     \
    --img_size 32     \
    --patch_size 4      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type spconv \
    --use_amp


python finetune_cifar_classifier_vit_jepa_repvgg_v2.py \
    --img_size 32     \
    --patch_size 8      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp


python finetune_cifar_classifier_vit_jepa_repvgg_v2.py \
    --pretrained_path ../../autodl-tmp/output_jepa_repvgg_v2_cifar/checkpoint_epoch_20.pth     \
    --img_size 32     \
    --patch_size 8      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp




python finetune_cifar_classifier_vit_jepa_repvgg.py \
    --img_size 128     \
    --patch_size 32      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp




python finetune_cifar_classifier_vit_jepa_repvgg.py \
    --pretrained_path ../../autodl-tmp/output_jepa_repvgg_sparse_in100_32p/checkpoint_epoch_200.pth     \
    --img_size 128     \
    --patch_size 32      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --use_amp


python finetune_cifar_classifier_vit_mae_cnn_spconv.py \
    --pretrained_path ../../autodl-tmp/output_mae_cnn_spconv_p32_in100/checkpoint_epoch_180.pth     \
    --img_size 128     \
    --patch_size 32      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type spconv \
    --use_amp



python finetune_cifar_classifier_vit_jepa_cnn_spconv.py \
    --pretrained_path ../../autodl-tmp/output_jepa_spconv_in100_32p/checkpoint_epoch_180.pth     \
    --img_size 128     \
    --patch_size 32      \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_type spconv \
    --use_amp


python finetune_ctrl_ldr_knn.py \
    --latent_dim 384     \
    --use_amp     \
    --pretrained_path ../../autodl-tmp/output_ecctrl/ebm_gan_checkpoint_700.pth
    