#!/bin/bash
# export paragrahvec model for flavordb
python export_vectors.py start --data_file_name 'flavordb_ver3.0.csv' --model_file_name 'flavordb_ver3.0_model.dm.sum_contextsize.3_numnoisewords.3_vecdim.300_batchsize.64_lr.0.001000_epoch.46_loss.1.421207_embed.1.pth.tar' --use_embeddings True
