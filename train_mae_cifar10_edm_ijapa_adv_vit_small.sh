python train_mae_cifar10_edm_ijepa_adv.py \
--use_amp \
--output_dir ../../autodl-tmp/output_edm_ijepa_adv_vit_small_imagenet100/ \
--img_size 224 \
--patch_size 16 \
--dataset 'tiny-imagenet' \
--data_path '../../autodl-tmp/imagenet100/' \
--save_freq 50 \
--embed_dim 384 \
--decoder_embed_dim 384 \
--num_heads 6 \
--decoder_num_heads 6 \
--mlp_ratio 4 \
--depth 12 \
--decoder_depth 4 \
--use_amp 