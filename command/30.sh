#!/bin/bash
# First argument is gpu number
# Second argument is gpu name

if [ -z "$1" ]
    then
        echo "Please enter gpu number as first argument!"
        exit 1
fi

if [ -z "$2" ]
    then
        echo "Please enter gpu name as second argument"
        exit 1
fi

gpu_num=$1
gpu_name=$2
theano_flags_device=gpu${gpu_num}

# Check whether gpu_name contains nscc
# If yes, do not specify GPU number in THEANO_FLAGS
# If no, specify GPU number in THEANO_FLAGS

if [[ $gpu_name == *"nscc"* ]]
    then
        theano_flags_device=gpu
fi

echo "Running script on ${theano_flags_device} : ${gpu_name}"

expt_num="31"
vocab_size="50000"
embedding_size="100"
word_embedding="glove.6B.100d"

model_type="crnn"
cnn_dim="100"
cnn_win="3"
cnn_layer="1"
rnn_type="lstm"
rnn_dim="100"
rnn_layer="1"
pooling_type="meanot"

optimizer="rmsprop"
num_epoch="25"
batch_size="32"
batch_eval_size="192"
dropout="0.5"
train_maxlen="0"

for rand in {1..5}
do
    CUDA_VISIBLE_DEVICES=${gpu_num} python train.py \
    -tr data/aclImdb/train/ -tu data/aclImdb/valid/ -ts data/aclImdb/test/ \
    --emb word2vec/vectors/${word_embedding}.txt \
    -o expt${expt_num}${gpu_num}-emb${embedding_size}-${word_embedding}-p${pooling_type}-sentiment-seed${rand}${gpu_num}78-${gpu_name} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} -trll ${train_maxlen} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78 --shuffle-seed ${rand}${gpu_num}78 \
    -nth
done

expt_num="32"
word_embedding="imdb_100d.w2v"
pooling_type="meanot"
for rand in {1..5}
do
    CUDA_VISIBLE_DEVICES=${gpu_num} python train.py \
    -tr data/aclImdb/train/ -tu data/aclImdb/valid/ -ts data/aclImdb/test/ \
    --emb word2vec/vectors/${word_embedding}.txt \
    -o expt${expt_num}${gpu_num}-emb${embedding_size}-${word_embedding}-p${pooling_type}-sentiment-seed${rand}${gpu_num}78-${gpu_name} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} -trll ${train_maxlen} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78 --shuffle-seed ${rand}${gpu_num}78 \
    -nth
done


expt_num="33"
word_embedding="glove.6B.100d"
pooling_type="att"
for rand in {1..5}
do
    CUDA_VISIBLE_DEVICES=${gpu_num} python train.py \
    -tr data/aclImdb/train/ -tu data/aclImdb/valid/ -ts data/aclImdb/test/ \
    --emb word2vec/vectors/${word_embedding}.txt \
    -o expt${expt_num}${gpu_num}-emb${embedding_size}-${word_embedding}-p${pooling_type}-sentiment-seed${rand}${gpu_num}78-${gpu_name} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} -trll ${train_maxlen} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78 --shuffle-seed ${rand}${gpu_num}78 \
    -nth
done

expt_num="34"
word_embedding="imdb_100d.w2v"
pooling_type="att"
for rand in {1..5}
do
    CUDA_VISIBLE_DEVICES=${gpu_num} python train.py \
    -tr data/aclImdb/train/ -tu data/aclImdb/valid/ -ts data/aclImdb/test/ \
    --emb word2vec/vectors/${word_embedding}.txt \
    -o expt${expt_num}${gpu_num}-emb${embedding_size}-${word_embedding}-p${pooling_type}-sentiment-seed${rand}${gpu_num}78-${gpu_name} \
    -t ${model_type} -p ${pooling_type} \
    -cl ${cnn_layer} -c ${cnn_dim} -w ${cnn_win} \
    -rl ${rnn_layer} -u ${rnn_type} -r ${rnn_dim} \
    --epochs ${num_epoch} -a ${optimizer} -e ${embedding_size} -v ${vocab_size} -do ${dropout} -trll ${train_maxlen} \
    -b ${batch_size} -be ${batch_eval_size} --seed ${rand}${gpu_num}78 --shuffle-seed ${rand}${gpu_num}78 \
    -nth
done

