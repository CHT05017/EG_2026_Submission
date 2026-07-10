CUDA_VISIBLE_DEVICES=7 python src/train.py --multi_gpus False --scaler 8 --batch_size 32 --log_dir './DIV2K_LOGS'
CUDA_VISIBLE_DEVICES=7 python src/train.py --multi_gpus False --scaler 4 --batch_size 32 --log_dir './DIV2K_LOGS'
CUDA_VISIBLE_DEVICES=7 python src/train.py --multi_gpus False --scaler 16 --batch_size 32 --log_dir './DIV2K_LOGS'
