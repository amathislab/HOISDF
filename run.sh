# python main/train.py --run_dir_name only_handobj_sdflist_witheval
# python main/train.py --run_dir_name gtsdf_affinity
# python sdf_nets/compute_scales.py

# python main/test_epochs.py --ckpt_path HFL_mano_cross_bigenc_6dec_ho3d_70 --dec_layers 6  --use_big_decoder --end_epoch 70 --point_sampling_epoch 40 --test_set test

python main/test.py --ckpt_path ckpts/dexycb_full/snapshot_dexycb_full.pth.tar > dexycb_full.out 2>&1

# python main/eval_dexycb.py --ckpt_path /mnt/haozhe_amg3/model_dump/objdirect_dexycb_smallep_pointsampling_smalldist/snapshot_129_193.pth.tar --dec_layers 6 --test_set test
# python main/eval_dexycb.py --ckpt_path /mnt/haozhe_amg3/model_dump/objdirect_dexycb_smallep_pointsampling_smalldist/snapshot_133_193.pth.tar --dec_layers 6 --test_set test
# python main/eval_dexycb.py --ckpt_path /mnt/haozhe_amg3/model_dump/objdirect_dexycb_smallep_pointsampling_smalldist/snapshot_148_193.pth.tar --dec_layers 6 --test_set test
