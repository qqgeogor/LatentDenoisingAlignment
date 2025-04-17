
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
