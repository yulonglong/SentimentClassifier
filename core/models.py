import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import math

import logging
logger = logging.getLogger(__name__)

def get_optimizer(args, params):
    """
    Get the optimizer class from PyTorch depending on the argument specified
    """
    
    import torch.optim as optim
    if args.algorithm == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.rmsprop_learning_rate, alpha=0.9, eps=1e-06, weight_decay=0, momentum=0, centered=False)
    elif args.algorithm == 'adam':
        optimizer = optim.Adam(params, lr=args.rmsprop_learning_rate)
    return optimizer

def load_embedding_reader(args):
    """
    Load embedding_reader object from binary file
    or
    Create embedding_reader object from the text file containing the embedding numbers for each word (word2vec output)
    """
    import pickle as pk
    emb_reader = None
    if args.emb_binary_path:
        from core.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info("Loading binary embedding data...")
        with open(args.emb_binary_path, 'rb') as emb_data_file:
            emb_reader = pk.load(emb_data_file)
        logger.info("Loading binary embedding data completed!")
    else:
        if args.emb_path:
            from core.w2vEmbReader import W2VEmbReader as EmbReader
            logger.info("Loading embedding data...")
            emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
            # if the seed starts with "10", which means first iteration and gpu num 0, then save
            if (args.seed / 100 == 10):
                with open(args.out_dir_path + '/data/emb_reader_instance_'+ str(args.emb_dim) +'.pkl', 'wb') as emb_data_file:
                    pk.dump(emb_reader, emb_data_file) # Note that saving an extremely big file/vocabulary will result in pickle memory error
            logger.info("Loading embedding data completed!")
    return emb_reader

class Attention(nn.Module):
    """Attention layer - Custom layer to perform weighted average over the second axis (axis=1)
        Transforming a tensor of size [N, W, H] to [N, 1, H].
        N: batch size
        W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        H: hidden state dimension or word embedding dimension
    Args:
        dim: The dimension of the word embedding
    Attributes:
        w: learnable weight matrix of size [dim, dim]
        v: learnable weight vector of size [dim]
    Examples::
        >>> m = models_pytorch.Attention(300)
        >>> input = Variable(torch.randn(4, 128, 300))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.att_weights = None
        self.w = nn.Parameter(torch.Tensor(dim, dim))
        self.v = nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, input):
        wplus = torch.mm(input.contiguous().view(-1, input.size()[2]), self.w)
        wplus = wplus.contiguous().view(-1, input.size()[1], self.w.size()[1])
        wplus = torch.tanh(wplus)

        att_w = torch.mm(wplus.contiguous().view(-1, wplus.size()[2]), self.v.contiguous().view(self.v.size()[0], 1))
        att_w = att_w.contiguous().view(-1, wplus.size()[1])
        att_w = F.softmax(att_w,dim=1)

        # Save attention weights to be retrieved for visualization
        self.att_weights = att_w

        after_attention = torch.bmm(att_w.unsqueeze(1), input)

        return after_attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + '1' + ', ' \
            + str(self.dim) + ')'

class ListModule(object):
    """
    A class to contain nn.Module inside nn.Module
    In this case it is used to store multiple cnn and rnn layers
    """
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class Net(nn.Module):
    def __init__(self, args, vocab, emb_reader):
        """
        :param args:
            the arguments from the main function containing all the configuration options
        :param vocab:
            vocabulary mapping from word to indices to initialize pre-trained word embeddings
        :param emb_reader:
            Embedding reader class which handles initialization of pre-trained word embeddings
        """
        super(Net, self).__init__()
        self.is_debug = False
        self.batch_number = None
        self.model_type = args.model_type
        self.pooling_type = args.pooling_type
        self.dropout_rate = args.dropout_rate
        self.att_weights = None # Attribute to save attention weights

        self.lookup_table = nn.Embedding(args.vocab_size, args.emb_dim)
        if emb_reader:
            self.init_embedding(vocab, emb_reader)
        
        self.cnn = None
        if self.model_type == 'cnn' or self.model_type == 'crnn' or self.model_type == 'crcrnn':
            self.cnn = ListModule(self, 'cnn_')
            for i in range(args.cnn_layer):
                self.cnn.append(nn.Conv2d(in_channels=1,
                            out_channels=args.cnn_dim,
                            kernel_size=(args.cnn_window_size, args.emb_dim),
                            padding=(args.cnn_window_size//2, 0)) # padding is on both sides, so padding=1 means it adds 1 on the left and 1 on the right
                )
        
        bidirectional = False
        if (args.is_bidirectional): bidirectional = True
        self.rnn = None
        if self.model_type == 'rnn' or self.model_type == 'crnn' or self.model_type == 'crcrnn':
            self.rnn = ListModule(self, 'rnn_')
            for i in range(args.rnn_layer):
                self.rnn.append(nn.LSTM(args.cnn_dim, args.rnn_dim, bidirectional=bidirectional))

        self.attention = None
        if self.pooling_type == 'att':
            self.attention = Attention(args.rnn_dim)

        self.linear = nn.Linear(args.rnn_dim, 1)

    def init_embedding(self, vocab, emb_reader):
        """
        Method to initialize lookup table using a pre-trained embedding
        """
        from core.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing lookup table...')
        initialized_weight = emb_reader.get_emb_matrix_given_vocab(vocab, self.lookup_table.weight.data.tolist())
        self.lookup_table.weight.data = torch.FloatTensor(initialized_weight)
        logger.info('Initializing lookup table completed!')

    def log(self, logString, tensor):
        logger.info(logString + str(tensor.size()))

    def tensorLogger(self, method, result_tensor):
        """
        :param method:
            The method or the layer name to be printed with the tensor
        :param result_tensor:
            The result tensor to be printed
        :return:
            Returns the same result_tensor
        """

        # result_tensor = method(tensor)
        layer_name = ""
        if isinstance(method, str):
            layer_name = method
        else:
            layer_name = method.__class__.__name__
        
        if self.is_debug and self.batch_number == 0: self.log('%15s :' % layer_name, result_tensor)
        return result_tensor

    def lstmWrapper(self, methodLstm, inputLstm):
        """
        A wrapper for LSTM because the odd input dimension/size.
        LSTM in PyTorch takes input of size [W, N, H]
        where W, N, H, are the number of words, batch size, and hidden states respectively
        http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch
        """
        bef_recc = self.tensorLogger("permute(1,0,2)",  inputLstm.permute(1, 0, 2))
        recc, hn = methodLstm(bef_recc)
        self.tensorLogger("LSTM", recc)
        self.tensorLogger("LSTM hn", hn[0])
        recc     = self.tensorLogger("permute(1,0,2)",  recc.permute(1, 0, 2))
        return recc

    def convWrapper(self, methodConv, inputConv):
        """
        A wrapper for Convplution layer because of the need to manipulate the dimension of input and output tensor
        Using convolution 2D requires some squeezing and unsqueezing
        """
        conv = self.tensorLogger("unsqueeze",       inputConv.unsqueeze(1))
        conv = self.tensorLogger(methodConv,        methodConv(conv))
        conv = self.tensorLogger("squeeze",         conv.squeeze())
        conv = self.tensorLogger("permute(0,2,1)",  conv.permute(0, 2, 1))
        return conv

    def forward(self, sentence, training=False, batch_number=None):
        """
        :param sentence:
                input sentence is in size of [N, W]
                N: batch size
                W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        :param training:
                boolean value, whether the forward is for training purpose
        :param batch_number:
                The current batch number
        :return:
                a tensor [C], where C is the number of classes
        """
        self.batch_number = batch_number

        self.tensorLogger("Initial", sentence)
        embed    = self.tensorLogger(self.lookup_table, self.lookup_table(sentence))
        conv     = embed
        
        
        if self.model_type == 'cnn' or self.model_type == 'crnn':
            for curr_cnn in self.cnn:
                prevConv = conv
                conv     = self.convWrapper(curr_cnn, conv)
                conv     = self.tensorLogger("dropout",         F.dropout(conv, p=self.dropout_rate, training=training))
            
        recc     = conv

        if self.model_type == 'rnn' or self.model_type == 'crnn':
            for curr_rnn in self.rnn:
                prevRecc = recc
                recc     = self.lstmWrapper(curr_rnn, recc)
                recc     = self.tensorLogger("dropout",         F.dropout(recc, p=self.dropout_rate, training=training))

        if self.model_type == 'crcrnn':
            assert (len(self.cnn) == len(self.rnn))
            for i in range(len(self.cnn)):
                prevConv = conv
                conv     = self.convWrapper(self.cnn[i], conv)
                conv     = self.tensorLogger("dropout",         F.dropout(conv, p=self.dropout_rate, training=training))
                conv     = self.lstmWrapper(self.rnn[i], conv)
                conv     = self.tensorLogger("dropout",         F.dropout(conv, p=self.dropout_rate, training=training))
            recc = conv

        if self.pooling_type == 'att':
            pool      = self.tensorLogger(self.attention,    self.attention(recc))
            self.att_weights = self.attention.att_weights # Save attention weights
        else:
            pool      = self.tensorLogger("Mean over time",  F.avg_pool2d(recc, (recc.size()[1],1)))
        pool      = self.tensorLogger("squeeze",         pool.squeeze(1))

        outlinear = self.tensorLogger(self.linear,     self.linear(pool))
        pred_prob = self.tensorLogger("sigmoid",        torch.sigmoid(outlinear))
        pred_prob = self.tensorLogger("squeeze",        pred_prob.squeeze())
        return pred_prob
