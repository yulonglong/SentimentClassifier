#!/bin/bash

# Negative Review
python test.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/neg/2710_1.txt

# Positive Review
python test.py \
-v best_model/vocab_v50000.pkl \
-m best_model/best_model_weights.h5 \
-ts data/aclImdb/test/pos/8580_10.txt

