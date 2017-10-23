#!/bin/bash
# Script to download and extract GloVe pre-trained embeddings

if [ ! -f glove.6B.zip ]; then
    echo "Downloading dataset from the internet..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
fi

echo "Extracting archive..."
unzip glove.6B.zip

echo "Word vectors preparation completed!"
