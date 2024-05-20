CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1  ft_curriculum_EPD_ddp.py --config configs/era5-EPD-6432-stage2.yml
