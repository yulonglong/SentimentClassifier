#!/bin/bash
# Script to train our own word embedding from our own corpus
# First argument is word embedding size

if [ -z "$1" ]
    then
        echo "Please enter the word embedding size"
        exit 1
fi

embedding_size=$1
combined_text_name="temp_combinedReviews.txt"
combined_tokenized_text_name="temp_combinedReviews_tokenized.txt"

# If w2v embedding file is not present
if [ ! -f word2vec/vectors/imdb_${embedding_size}d.w2v.txt ]; then
    # If tokenized file is not present
    if [ ! -f word2vec/vectors/${combined_tokenized_text_name} ]; then
        # If combined text file is not present
        if [ ! -f word2vec/vectors/${combined_text_name} ]; then
            echo "Combining dataset into one text file..."
            # Compile the combine text script
            g++ word2vec/combineText4word2vec.cpp -o word2vec/combineText4word2vec.out
            # Execute combine text script
            ./word2vec/combineText4word2vec.out data/aclImdb/train/unsup/ word2vec/vectors/${combined_text_name}
        fi
        echo "Tokenizing text..."
        # Tokenize text
        python3 core/tokenizeText.py -in word2vec/vectors/${combined_text_name} -out word2vec/vectors/${combined_tokenized_text_name}
    fi
    # Train word2vec
    echo "Training word2vec"
    ./word2vec/word2vec -train word2vec/vectors/${combined_tokenized_text_name} -output word2vec/vectors/imdb_${embedding_size}d.w2v.txt -size ${embedding_size} -hs 1 -cbow 0 -threads 6
fi
echo "Done!"
