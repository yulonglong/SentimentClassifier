from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
import core.utils as U
import core.helper as helper
import pickle as pk
import os.path

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vocab-path", dest="vocab_path", type=str, metavar='<str>', required=True, help="The path to the vocab file")
parser.add_argument("-m", "--model-path", dest="model_path", type=str, metavar='<str>', required=True, help="The path to the model file")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test file")

args = parser.parse_args()

U.set_logger()
U.print_args(args)

import core.reader as dataset_reader
from core.evaluator import Evaluator

#######################################################################################
## Prepare data
#
t0 = time()

logger.info("Loading vocab...")
with open(args.vocab_path, 'rb') as vocab_file:
	[vocab] = pk.load(vocab_file)

logger.info("Loading vocab completed!")

test_x = dataset_reader.read_dataset_single(args.test_path, vocab, True, True)
# make duplicate of the same set so it doesnt get squeezed in the model if batch size is 1
test_x = np.array([test_x, test_x]).squeeze()

# Get just the filename without path and trailing file format
base = os.path.basename(args.test_path)
filename = os.path.splitext(base)[0]
filename_list = [filename, filename] # make duplicate to match the test_x format

######################################################################################################
## Load model
#

import torch
from torch.autograd import Variable

logger.info("Loading model...")
model = torch.load(args.model_path, map_location={'cuda:0': 'cpu'})
logger.info("Loading model completed!")
# logger.info(model)

######################################################################################################
## Testing
#
t1 = time()

pytorch_test_x = torch.from_numpy(test_x.astype('int64'))
pytorch_test_x = Variable(pytorch_test_x)
outputs = model(pytorch_test_x)
outputs = outputs.cpu().data.numpy()

percentScore = outputs[0] * 100.0
if (percentScore >= 50.0): 
	logger.info(U.BColors.BOKGREEN + "Prediction Score: %.2f %% (Positive Review)" % (percentScore))
else:
	logger.info(U.BColors.BRED + "Prediction Score: %.2f %% (Negative Review)" % (percentScore))


#############################################################################################
## Start Attention Visualization
#

logger.info("Processing attention visualization...")

# Create output folder
output_foldername = 'test_single_output/'
U.mkdir_p(output_foldername)

# make duplicate to match the test_x format
score_list = [percentScore, percentScore]

# Get the attention weights from the model
attention_weights = model.att_weights.data.numpy()
assert (test_x.shape == attention_weights.shape)

# Do attention visualization and export to pdf
helper.do_attention_visualization(attention_weights, test_x, vocab, filename_list, score_list, output_foldername=output_foldername)

#############################################################################################

preparation_time = t1 - t0
preparation_time_minutes = preparation_time/60
prediction_time = time() - t1
prediction_time_minutes = prediction_time/60
total_time = time() - t0
total_time_minutes = total_time/60

logger.info('------------------------------------------------------------------------')
logger.info('Preparation time : %i seconds in total (%.1f minutes)' % (preparation_time, preparation_time_minutes))
logger.info('Prediction time  : %i seconds in total (%.1f minutes)' % (prediction_time, prediction_time_minutes))
logger.info('Total time       : %i seconds in total (%.1f minutes)' % (total_time, total_time_minutes))
logger.info('------------------------------------------------------------------------')
