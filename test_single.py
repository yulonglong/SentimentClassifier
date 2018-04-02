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

pred_string = None
percentScore = outputs[0] * 100.0
if (percentScore >= 50.0): 
	logger.info(U.BColors.BOKGREEN + "Prediction Score: %.2f %% (Positive Review)" % (percentScore))
	pred_string = "positive"
else:
	logger.info(U.BColors.BRED + "Prediction Score: %.2f %% (Negative Review)" % (percentScore))
	pred_string = "negative"


#############################################################################################
## Start Attention Visualization
#

logger.info("Processing attention visualization...")

# Create output folder
output_foldername = 'test_single_output/'
U.mkdir_p(output_foldername)

# Get just the filename without path and trailing file format
base = os.path.basename(args.test_path)
filename = os.path.splitext(base)[0]

# Get reverse vocab
def create_reverse_vocab(vocab):
	reverse_vocab = {0:'<pad>', 1:'<unk>', 2:'<num>', 3:'<newline>'}
	for word, index in vocab.iteritems():
		reverse_vocab[index] = word
	return reverse_vocab

reverse_vocab = create_reverse_vocab(vocab)

# Get the attention weights from the model
attention_weights = model.att_weights.data.numpy()
assert (test_x.shape == attention_weights.shape)

num_essays = test_x.shape[0]
num_words = test_x.shape[1]

# Import all the necessary libraries for plotting
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Loop the number of essays/reviews so that we can reuse the code for the batch processing
for i in range(num_essays):
	curr_filename = filename # Only one file, so use the original filename
	scaled_pred = percentScore # no scaled_prediction in this case, need to modify when threshold is not 0.5

	word_sequence = []
	attention_weights_normalized = []
	attention_weights_max = 0.0
	# Get the max value of the attention weights
	for j in range(num_words):
		if attention_weights_max < attention_weights[i][j]:
			attention_weights_max = attention_weights[i][j]
	# Encode the normalized weights for the words ignoring the padding
	for j in range(num_words):
		current_word = reverse_vocab[test_x[i][j]].encode('utf-8', 'ignore')
		if current_word == '<pad>':
			continue
		word_sequence.append(current_word)
		attention_weights_normalized.append(attention_weights[i][j]/float(attention_weights_max))

	assert len(word_sequence) == len(attention_weights_normalized)
	current_num_words = len(word_sequence)

	# Create PDF document
	pdf_pages = PdfPages(output_foldername +'/' + str(curr_filename) + '.pdf')
	j = 0
	done = False
	page = 0

	while not done:
		# A4 size to be printed on paper, dpi somehow cant increase beyond 100, therefore increasing the plot size
		# All numbers are magic numbers according to trial and error
		fig = plt.figure(figsize=(8.27*3, 11.69*3), dpi=100)
		curr_x = 0.005
		curr_y = 0.98
		plt.axes([0.025,0.025,0.95,0.95])
		plt.axis('off')
		while j < current_num_words:
			word = word_sequence[j]
			# If new line, print new line and continue
			if word == '<newline>':
				curr_x = 0.005
				curr_y -= 0.02
				j += 1
				continue

			# print words with the background colour
			word_len = len(word)
			t = plt.text(curr_x, curr_y, word, ha='left', va='center', color="#000000", alpha=1.0,
					transform=plt.gca().transAxes, fontsize=30)
			t.set_bbox(dict(color='red', alpha=attention_weights_normalized[j]))

			# increase xy coordinate for the next text
			curr_x += (word_len**(0.75)) *0.022
			if (curr_x > 0.90):
				curr_x = 0.005
				curr_y -= 0.02

			j += 1

			if (curr_y < 0.01):
				break

		# Show in the picture the prediction value and the filename
		if page == 0:
			ymin, ymax = plt.ylim()
			xmin, xmax = plt.xlim()
			plt.text(
				xmin, ymax, 'ID : %s     Prediction : %s    Prediction score: %.4f%%' %
				(curr_filename, pred_string, scaled_pred) , fontsize=30, style='normal',
				bbox={'facecolor':'yellow', 'alpha':1.0, 'pad':10})

		pdf_pages.savefig(fig)
		plt.close(fig)
		page += 1

		if j == current_num_words: done = True
	
	if page > 1:
		logger.info("        Number of pages : " + str(page))
	pdf_pages.close()

logger.info(U.BColors.BOKGREEN + 'Attention visualization is successful! Please find the pdf in ' + output_foldername + filename + '.pdf')

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
