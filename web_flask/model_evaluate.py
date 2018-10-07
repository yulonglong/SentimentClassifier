from core import reader as dataset_reader
from core.evaluator import Evaluator
from core import utils as U
import core.utils as U
import core.helper as helper
import pickle as pk
import torch
from torch.autograd import Variable
import numpy as np

from datetime import datetime

import logging
logger = logging.getLogger(__name__)
global_base_path = "web_flask/web_movie_review/"
U.mkdir_p(global_base_path)
U.set_logger(global_base_path)

class ModelEvaluate:
	def __init__(self):
		self.base_path = global_base_path
		self.vocab_path = "best_model/vocab_v50000.pkl"
		self.model_path = "best_model/best_model_weights.h5"

		#######################################################################################
		## Prepare data
		#
		logger.info("Loading vocab...")
		with open(self.vocab_path, 'rb') as vocab_file:
			[self.vocab] = pk.load(vocab_file)
		logger.info("Loading vocab completed!")

		######################################################################################################
		## Load model
		#
		logger.info("Loading model...")
		self.model = torch.load(self.model_path, map_location={'cuda:0': 'cpu'})
		logger.info("Loading model completed!")
		# logger.info(model)

	def load_dataset(self, filename):
		test_path = self.base_path + filename + ".txt"
		test_x = dataset_reader.read_dataset_single(test_path, self.vocab, True, True)
		# make duplicate of the same set so it doesnt get squeezed in the model if batch size is 1
		test_x = np.array([test_x, test_x]).squeeze()
		filename_list = [test_path, test_path] # make duplicate to match the test_x format
		return test_x, filename_list

	def write_to_file(self, movie_review):
		curr_ms = str(datetime.now().microsecond) # For accuracy in multiple request setting
		curr_time = str(datetime.today().strftime('%Y-%m-%d_%H-%M-%S')) # Getting current date for filename
		curr_time += "_" + curr_ms
		with open(self.base_path + curr_time + ".txt", "w") as outfile:
			outfile.write(movie_review)
		return curr_time

	def attention_visualization(self, percentScore, test_x, filename_list):
		#############################################################################################
		## Start Attention Visualization
		#
		logger.info("Processing attention visualization...")
		# make duplicate to match the test_x format
		score_list = [percentScore, percentScore]
		# Get the attention weights from the model
		attention_weights = self.model.att_weights.data.numpy()
		# assert (test_x.shape == attention_weights.shape)
		# Do attention visualization and export to pdf
		helper.do_attention_visualization(attention_weights, test_x, self.vocab, filename_list, score_list, output_foldername=self.base_path)
		logger.info("Attention visualization completed!")

	def evaluate(self, movie_review):
		# 1. Write to file
		curr_time = self.write_to_file(movie_review)
		# 2. Load and preprocess text file
		test_x, filename_list = self.load_dataset(curr_time)
		# 3. Evaluate using model
		pytorch_test_x = torch.from_numpy(test_x.astype('int64'))
		pytorch_test_x = Variable(pytorch_test_x)
		outputs = self.model(pytorch_test_x)
		outputs = outputs.cpu().data.numpy()
		percentScore = outputs[0] * 100.0
		if (percentScore >= 50.0): 
			logger.info(U.b_green("Prediction Score: %.2f %% (Positive Review)" % (percentScore)))
		else:
			logger.info(U.b_red("Prediction Score: %.2f %% (Negative Review)" % (percentScore)))
		self.attention_visualization(percentScore, test_x, filename_list)
		
		pdf_filename = curr_time + ".pdf"
		return percentScore, pdf_filename
