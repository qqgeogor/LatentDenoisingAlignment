# python train_vit_ldae_dino.py --use_amp --output_dir ../../autodl-tmp/output_tiny/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train'

python train_vit_ldae_dino.py --use_amp --output_dir ../../autodl-tmp/output_tiny/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train' --num_views 2

# python train_vit_ldae_dino.py --use_amp --output_dir /mnt/d/repo/vit_ldae_tiny/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '/mnt/d/datasets/tiny-imagenet-200/train' --num_views 2


# python train_vit_ldae_dino_self_dist.py --use_amp --output_dir ../../autodl-tmp/output_tiny_self_dist/  --img_size 128 --patch_size 16 --dataset 'tiny-imagenet' --data_path '../../autodl-tmp/tiny-imagenet-200/train'


