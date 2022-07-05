# PYTHONPATH="./:${PYTHONPATH}" python scripts/data_preparation/download_datasets.py
# PYTHONPATH="./:${PYTHONPATH}" python scripts/data_preparation/create_lmdb2.py

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/train.py -opt options/train/LPTN/train_FiveK.yml
# NCCL_DEBUG=INFO PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/LPTN/train_FiveK.yml --launcher pytorch
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/LPTT/train_FiveK.yml --launcher pytorch

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/train_FiveK_03.yml --launcher pytorch

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/train_FiveK_05.yml --launcher pytorch
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 codes/train.py -opt options/train/LPTN/train_FiveK.yml --launcher pytorch

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/train_FiveK_11.yml --launcher pytorch
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 codes/train.py -opt options/train/LPTN/train_FiveK.yml --launcher pytorch

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/test.py -opt options/test/LPTN/test_FiveK.yml
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/test_speed.py -opt options/test/LPTT/test_speed_FiveK.yml

# CUDA_VISIBLE_DEVICES=3 python inference.py

cp experiments/LPTT_FiveK_L3_PEG-1k9/models/net_g_latest.pth \
    experiments/LPTT_FiveK_L4_PEG-1k9/models/net_g_latest.pth \
    experiments/LPTT_FiveK_L5_PEG-1k5/models/net_g_latest.pth \
    experiments/LPTT_FiveK_L6_PEG-1k3/models/net_g_latest.pth \

cp experiments/LPTT_FiveK_L6_PEG-1k3/models/net_g_latest.pth pretrained/net_g_latest_L6.pth 