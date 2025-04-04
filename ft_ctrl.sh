
python finetune_ctrl.py     \
    --pretrained_path /mnt/d/repo/output/cifar10-ebm-gan-r3gan-ctrl/ebm_gan_checkpoint_160.pth     \
    --img_size 32 \
    --use_amp 



# python finetune_ctrl_vit.py     \
#     --pretrained_path /mnt/d/repo/output/r3gan-ctrl-vit/ebm_gan_checkpoint_140.pth     \
#     --img_size 32 \
#     --use_amp \
#     --freeze_backbone 