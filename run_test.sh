#!/bin/bash

echo "(run_test.sh): Starting test on a single negative review..."
# Negative Review
python3 test_single.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/neg/2710_1.txt

echo "(run_test.sh): Starting test on a single positive review..."
# Positive Review
python3 test_single.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/pos/8580_10.txt

echo "(run_test.sh): Starting test on 50,000 reviews..."
echo "(run_test.sh): This is going to take a while, please be patient..."
echo "(run_test.sh): If you would like to visualize attention, please add '-a' argument in the command"
echo "(run_test.sh): Note that you need a lot of computing power to create 50,000 PDF attention visualization"
# Batch Review
python3 test_batch.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-b 1024 \
-ts data/aclImdb/train/unsup/
