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
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=192, help="Batch size for testing (default=192). Increase the value for faster processing if you have larger RAM size.")

args = parser.parse_args()

output_foldername = 'test_batch_output/'
U.mkdir_p(output_foldername)
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

logger.info("Loading dataset...")
# test_x, test_y, test_filename_y, _, _, _, _ = dataset_reader.read_dataset_folder(args.test_path + "/*", 0, vocab, True, True)
datasetMulti = dataset_reader.ReadDatasetFolder(args.test_path + "/*", 0, vocab, True, True)
test_x, test_y, test_filename_y, _, _, _, _ = datasetMulti.read_dataset_multithread()

test_x, test_y, test_filename_y, _ = (
    helper.sort_and_split_data_into_chunks(
        test_x, test_y, test_filename_y, args.batch_size)
)
logger.info("Loading dataset completed!")

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
test_pred = np.array([])

logger.info("Predicting scores...")
for idx, _ in enumerate(test_x):
    pytorch_test_x = torch.from_numpy(test_x[idx].astype('int64'))
    curr_test_pred = model(Variable(pytorch_test_x))
    curr_test_pred = curr_test_pred.cpu().data.numpy()
    test_pred = np.append(test_pred, curr_test_pred)
logger.info("Prediction completed...")

# Sort the prediction result based on filename
assert len(test_pred) == len(test_filename_y)
filename_result, pred_result = (list(t) for t in zip(*sorted(zip(test_filename_y, test_pred))))

with open(output_foldername + "result.csv", "w") as outfile:
    assert len(filename_result) == len(pred_result)
    outfile.write("Filename,Prediction\n")
    for i in xrange(len(filename_result)):
        outfile.write('%s,%.4f\n' % (filename_result[i],pred_result[i]))
    logger.info(str(len(filename_result)) + " files has been reviewed successfully.")
    logger.info("Results are saved in " + U.BColors.BOKGREEN + output_foldername + "result.csv")

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
