#!/bin/bash
# Train paragrahvec model for flavordb
python train.py start --data_file_name 'flavordb_ver3.0.csv' --model_ver dm --context_size 3 --num_epochs 100 --batch_size 64 --num_noise_words 3 --vec_dim 300 --lr 1e-3 --num_workers -1 --use_embeddings True
