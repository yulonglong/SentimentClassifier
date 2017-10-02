#!/bin/bash
# Script to prepare dataset from scratch

# If w2v embedding file is not present
if [ ! -f aclImdb_v1.tar.gz ]; then
    echo "Downloading dataset from the internet..."
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
fi

echo "Extracting archive..."
tar -xzvf aclImdb_v1.tar.gz

echo "Preprocessing data with validation set"
javac Preprocess.java
java Preprocess

echo "Dataset preparation completed!"
