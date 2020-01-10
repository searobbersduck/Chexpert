# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/lse_cam.json logdir/lse_cam --num_workers 16 --device_ids "4,5,6,7"|tee logdir/lse_cam.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/lse_sam.json logdir/lse_sam --num_workers 16 --device_ids "4,5,6,7"|tee logdir/lse_sam.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/lse_fpa.json logdir/lse_fpa --num_workers 16 --device_ids "4,5,6,7"|tee logdir/lse_fpa.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/lse_none.json logdir/lse_none --num_workers 16 --device_ids "4,5,6,7"|tee logdir/lse_none.log

# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/avgmax_cam.json logdir/avgmax_cam --num_workers 16 --device_ids "4,5,6,7"|tee logdir/avgmax_cam.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/avgmax_sam.json logdir/avgmax_sam --num_workers 16 --device_ids "4,5,6,7"|tee logdir/avgmax_sam.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/avpmax_fpa.json logdir/avpmax_fpa --num_workers 16 --device_ids "4,5,6,7"|tee logdir/avpmax_fpa.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bin/train.py config/avgmax_none.json logdir/avgmax_none --num_workers 16 --device_ids "4,5,6,7"|tee logdir/avgmax_none.log

