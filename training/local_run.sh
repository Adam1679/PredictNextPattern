# deepspeed --num_nodes 1 --num_gpus=8 training/trainer_ds.py
deepspeed --num_nodes 1 --num_gpus=8 training/trainer_ds.py --eval_only --ckpt=checkpoints/