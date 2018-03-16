#!/bin/bash

echo "(run_test.sh): Starting test on a single negative review..."
# Negative Review
python test_single.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/neg/2710_1.txt

echo "(run_test.sh): Starting test on a single positive review..."
# Positive Review
python test_single.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/pos/8580_10.txt

echo "(run_test.sh): Starting test on 50,000 reviews..."
echo "(run_test.sh): This is going to take a while, please be patient..."
# Batch Review
python test_batch.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-b 1024 \
-ts data/aclImdb/train/unsup/
