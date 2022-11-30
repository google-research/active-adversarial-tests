export PYTHONPATH=./:$PYTHONPATH
checkpoint_dir=~/models/adv_interp_model/
mkdir -p $checkpoint_dir
CUDA_VISIBLE_DEVICES=0 python train_adv_interp.py \
    --resume \
    --lr=0.1 \
    --model_dir=$checkpoint_dir \
    --init_model_pass=-1 \
    --max_epoch=200 \
    --save_epochs=10 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=128 \
    --label_adv_delta=0.5 \
    --save_epochs=100 \

