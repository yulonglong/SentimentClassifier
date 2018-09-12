from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
from core import utils as U
from core import helper as helper
import pickle as pk
import os.path

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', default=None, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', default=None, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', default=None, help="The path to the test set")
parser.add_argument("-db", "--data-binary", dest="data_binary_path", type=str, metavar='<str>', default=None, help="The path to the processed data binary file")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='rnn', help="Model type (attnn|rnn|cnn|vdcnn|rcnn|crnn|cwrnn|cwresrnn|cnn+cnn|rnn+cnn) (default=rnn)")
parser.add_argument("-p", "--pooling-type", dest="pooling_type", type=str, metavar='<str>', default='meanot', help="Pooling type (meanot|att) (default=meanot)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, metavar='<float>', default=0.001, help="Learning rate in rmsprop (default=0.001)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-cl","--cnn-layer", dest="cnn_layer", type=int, metavar='<int>', default=1, help="Number of CNN layer (default=1)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=2, help="CNN window size. (default=2)")
parser.add_argument("-rl","--rnn-layer", dest="rnn_layer", type=int, metavar='<int>', default=1, help="Number of RNN layer (default=1)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size for training (default=32)")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=256, help="Batch size for evaluation (default=256)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("-do", "--dropout", dest="dropout_rate", type=float, metavar='<float>', default=0.5, help="The dropout rate in the model (default=0.5)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("-eb", "--emb-binary", dest="emb_binary_path", type=str, metavar='<str>', help="The path to the loaded embedding reader class instance")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")
parser.add_argument("--shuffle-seed", dest="shuffle_seed", type=int, metavar='<int>', default=1337, help="Random shuffle seed (default=1337)")
parser.add_argument("-bi", "--bidirectional", dest="is_bidirectional", action='store_true', help="Flag to enable bidirectional RNN (i.e., only process dataset)")

parser.add_argument("-trll", "--train-length-limit", dest="train_length_limit", type=int, metavar='<int>', default=0, help="The maximum length of any training instance, longer instances will get thrown away (default=0, use all)")
parser.add_argument("-sep", "--save-every-epoch", dest="is_save_every_epoch", action='store_true', help="Flag to save complete model weights every epoch (default=False)")

parser.add_argument("-nmt", "--no-multithread", dest="no_multithread", action='store_true', help="Flag to disable multithreading when reading dataset")
parser.add_argument("-nt", "--no-train", dest="no_train", action='store_true', help="Flag to disable training (i.e., only process dataset)")
parser.add_argument("-nth", "--no-threshold", dest="no_threshold", action='store_true', help="Flag not to do threshold adjustment (i.e., using 0.5 as threshold)")
parser.add_argument("-ncw", "--no-class-weight", dest="no_class_weight", action='store_true', help="Flag not to use class weight (i.e., not using weighted loss function based on class distribution)")
parser.add_argument("--shuffle-path", dest="shuffle_path", type=str, metavar='<str>', help="The path to the shuffle file containing the permutation of index for training (*.txt) (numpy array)")

args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/data')
U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.set_logger(out_dir)
U.print_args(args)

import sys
sys.stdout = open(out_dir + '/stdout.txt', 'w')
sys.stderr = open(out_dir + '/stderr.txt', 'w')

####################################################################################
## Argument Validation
#

# Assert to check either train,dev,test path or processed data path is specified
# Either of them must be set
assert ((args.data_binary_path) or (args.train_path and args.dev_path and args.test_path))

valid_model_type =  {'crnn', 'cnn', 'rnn', 'crcrnn'}
model_type_binary = {'crnn', 'cnn', 'rnn', 'crcrnn'}

assert args.model_type in valid_model_type
assert args.algorithm in {'rmsprop', 'adam'}
assert args.pooling_type in { 'meanot', 'att'}

from core import reader as dataset_reader
from core.evaluator import Evaluator

#######################################################################################
## Prepare data
#

((train_x, train_y, train_filename_y),
 (dev_x, dev_y, dev_filename_y),
 (test_x, test_y, test_filename_y),
 vocab, vocab_size, overal_maxlen) = dataset_reader.load_dataset(args)

# Initialize Embedding reader
from core.w2vEmbReader import load_embedding_reader
emb_reader = load_embedding_reader(args)

# Stop script here if no train is specified
if args.no_train:
    exit()

############################################################################################
## if truncate train length is specified, throw training instances longer than the length
#
if (args.train_length_limit > 0):
    new_train_x, new_train_y, new_train_filename_y = [], [], []
    for i in range(len(train_x)):
        if (len(train_x[i]) <= args.train_length_limit):
            new_train_x.append(train_x[i])
            new_train_y.append(train_y[i])
            new_train_filename_y.append(train_filename_y[i])
    train_x, train_y, train_filename_y = new_train_x, new_train_y, new_train_filename_y

############################################################################################
## Set numpy random seed
#

if args.seed > 0:
    logger.info('Setting np.random.seed(%d)' % args.seed)
    np.random.seed(args.seed)

#######################################################################################
## Create permutation of indexes for the training data and save them in a file
#

permutation_list = helper.get_permutation_list(args, train_y)

############################################################################################
## Initialize Evaluator
## WARNING! Must be initialize Evaluator before padding to do dynamic padding
#

evl = Evaluator(
    out_dir,
    (train_x, train_y, train_filename_y),
    (dev_x, dev_y, dev_filename_y),
    (test_x, test_y, test_filename_y), no_threshold=args.no_threshold, batch_size_eval=args.batch_size_eval)

############################################################################################
## Padding to dataset for statistics
#

# Pad sequences for mini-batch processing
padded_train_x = helper.pad_sequences(train_x)
padded_dev_x = helper.pad_sequences(dev_x)
padded_test_x = helper.pad_sequences(test_x)

############################################################################################
## Some statistics
#

bincount = np.bincount(train_y)
most_frequent_class = bincount.argmax()
np.savetxt(out_dir + '/preds/bincount.txt', bincount, fmt='%i')

np_train_y = np.array(train_y, dtype='float32')
np_dev_y = np.array(dev_y, dtype='float32')
np_test_y = np.array(test_y, dtype='float32')

train_mean = np_train_y.mean()
train_std = np_train_y.std()
dev_mean = np_dev_y.mean()
dev_std = np_dev_y.std()
test_mean = np_test_y.mean()
test_std = np_test_y.std()

logger.info('Statistics:')

logger.info('  train_x shape: ' + str(np.array(padded_train_x).shape))
logger.info('  dev_x shape:   ' + str(np.array(padded_dev_x).shape))
logger.info('  test_x shape:  ' + str(np.array(padded_test_x).shape))

logger.info('  train_y shape: ' + str(np_train_y.shape))
logger.info('  dev_y shape:   ' + str(np_dev_y.shape))
logger.info('  test_y shape:  ' + str(np_test_y.shape))

logger.info('  train_y mean: %.3f, stdev: %.3f, MFC: %i' % (train_mean, train_std, most_frequent_class))

############################################################################################
## Make sure the size of each mini-batch is more than 1 to prevent model crash
# If the assertions below fail, please change to batch size to another number
assert (np_train_y.shape[0] % args.batch_size != 1)
assert (np_train_y.shape[0] % args.batch_size_eval != 1)
assert (np_dev_y.shape[0] % args.batch_size_eval != 1)
assert (np_test_y.shape[0] % args.batch_size_eval != 1)

############################################################################################
## Compute class weight (where data is usually imbalanced)
# Always imbalanced in medical text data

class_weight = helper.compute_class_weight(np.array(train_y, dtype='float32'))

######################################################################################################
## Create model
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.models import Net

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

model = Net(args, vocab, emb_reader)
criterion = nn.BCELoss()
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

logger.info(model)

from core.models import get_optimizer
optimizer = get_optimizer(args, model.parameters())

total_train_time = 0
total_eval_time = 0

import copy
copy_train_x = copy.deepcopy(train_x)

logger.info('------------------------------------------------------------------------')
logger.info('Initial Evaluation:')
evl.evaluate(model, -1)

content = evl.print_info()

######################################################################################################
## Training
#

for ii in range(args.epochs):
    assert (len(train_x) == len(copy_train_x))

    # Get current set of permutation indexes
    curr_perm = permutation_list[ii]
    # Split data and dynamically pad based on the batch size
    perm_train_x, perm_train_y, _ = helper.sort_data_given_index(train_x, train_y, curr_perm)
    train_x_list, train_y_list, _, _ = helper.split_data_into_chunks(perm_train_x, perm_train_y, args.batch_size, combine_y=False)

    t0 = time()

    # Train in chunks of batch size and dynamically padded
    train_loss_sum = 0
    for idx, _ in enumerate(train_x_list):
        train_input = train_x_list[idx]

        pytorch_train_x = torch.from_numpy(train_x_list[idx].astype('int64'))
        pytorch_train_y = torch.from_numpy(train_y_list[idx].astype('float32'))

        # Create class weight tensor
        pytorch_weight = None
        if not args.no_class_weight:
            weight = copy.deepcopy(train_y_list[idx])
            for key, value in class_weight.items():
                indices = (weight == key)
                weight[indices] = value
            weight = np.asarray(weight, dtype=np.float32)
            pytorch_weight = torch.from_numpy(weight.astype('float32'))
        
        start_time = time()
        if torch.cuda.is_available():
            st = time()
            pytorch_train_x = pytorch_train_x.cuda()
            pytorch_train_y = pytorch_train_y.cuda()
            if not args.no_class_weight: pytorch_weight = pytorch_weight.cuda()
            dt = time() - st
            
        pytorch_train_x, pytorch_train_y = Variable(pytorch_train_x), Variable(pytorch_train_y)
        optimizer.zero_grad()
        outputs = model(pytorch_train_x, training=True, batch_number=idx)
        if not args.no_class_weight: criterion.weight = pytorch_weight # Assign weights to the training data
        loss = criterion(outputs, pytorch_train_y)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.data.item()

    train_loss = train_loss_sum / (len(train_x_list))

    tr_time = time() - t0
    total_train_time += tr_time

    # Evaluate
    t0 = time()
    # Only evaluate on validation and test set if train_loss is smaller than 1.0
    if (train_loss < 1.0): evl.evaluate(model, ii)
    evl_time = time() - t0
    total_eval_time += evl_time

    if args.is_save_every_epoch:
        # Save model and optimizer state for every epoch
        model.save(out_dir + '/models/model_complete_epoch'+ str(ii) +'.h5', overwrite=True)
        model.save_weights(out_dir + '/models/model_weights_epoch'+ str(ii) +'.h5', overwrite=True)
        with open(args.out_dir_path + '/models/model_np_random_state_epoch'+ str(ii) + '.pkl', 'wb') as np_random_file:
            pk.dump(np.random.get_state(), np_random_file)

    logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0))
    logger.info('[Train] loss: %.4f' % (train_loss))

    content = evl.print_info()

###############################################################################################################################
## Summary of the results
#

total_time = total_train_time + total_eval_time

total_train_time_hours = total_train_time/3600
total_eval_time_hours = total_eval_time/3600
total_time_hours = total_time/3600

logger.info('Training:   %i seconds in total (%.1f hours)' % (total_train_time, total_train_time_hours))
logger.info('Evaluation: %i seconds in total (%.1f hours)' % (total_eval_time, total_eval_time_hours))
logger.info('Total time: %i seconds in total (%.1f hours)' % (total_time, total_time_hours))
logger.info('------------------------------------------------------------------------')

content = evl.print_final_info()
