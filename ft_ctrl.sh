
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