#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/data2/rezDrive/videoMasking/MGMAE/output/vit_b_k400_mgmae_smol'
DATA_PATH='/data2/rezDrive/subset_k400/labels_smol.txt'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


        python -u run_mgmae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type mgmae \
        --mask_ratio 0.9 \
        --init_mask_map mix_gauss \
        --model pretrain_videomae_base_patch16_224 \
        --base_frame middle \
        --warp_type backward \
        --hole_filling consist \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --lr 1e-3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 20 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}