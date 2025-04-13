
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
    --pretrained_path ../../autodl-tmp/output_ctrl_mcr2/ebm_gan_checkpoint_900.pth 





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
    