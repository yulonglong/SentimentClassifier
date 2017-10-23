#!/bin/bash

# Negative Review
python test.py \
-v expt100-sentiment-seed1078-GTX1080/data/vocab_v50000.pkl \
-m expt100-sentiment-seed1078-GTX1080/models/best_model_weights.h5 \
-ts data/aclImdb/train/unsup/16088_0.txt

# Positive Review
python test.py \
-v expt100-sentiment-seed1078-GTX1080/data/vocab_v50000.pkl \
-m expt100-sentiment-seed1078-GTX1080/models/best_model_weights.h5 \
-ts data/aclImdb/train/unsup/74_0.txt

