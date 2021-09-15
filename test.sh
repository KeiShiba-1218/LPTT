# PYTHONPATH="./:${PYTHONPATH}" python scripts/data_preparation/download_datasets.py
# PYTHONPATH="./:${PYTHONPATH}" python scripts/data_preparation/create_lmdb2.py

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/train.py -opt options/train/LPTN/train_FiveK.yml
# NCCL_DEBUG=INFO PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/LPTN/train_FiveK.yml --launcher pytorch
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/LPTT/train_FiveK.yml --launcher pytorch
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 codes/train.py -opt options/train/LPTN/train_FiveK.yml --launcher pytorch

# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/test.py -opt options/test/LPTT/test_FiveK.yml
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python codes/test_speed.py -opt options/test/LPTT/test_speed_FiveK.yml