import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("initalize linear weight")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        #########################################
        # here you need to define the "self.rnn_lstm"  the input size is "embedding_dim" and the output size is "lstm_hidden_dim"
        # the lstm should have two layers, and the  input and output tensors are provided as (batch, seq, feature)
        # ! input with shape (seq, batch, feature)
        self.num_layers = 2
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=self.num_layers, batch_first=True
        )
        ##########################################

        self.relu = nn.ReLU()
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)  # call the weights initial function.

        self.softmax = nn.LogSoftmax(dim=-1)  # the activation function.
        # self.tanh = nn.Tanh()

    def forward(self, sentences):
        # input being list of lists (of varying length), convert to batch input shape: (batch, max_length, f)
        data_tensors = [torch.tensor(data, dtype=torch.long) for data in sentences]
        sentences = pad_sequence(data_tensors, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in data_tensors])
        batch_input = self.word_embedding_lookup(sentences).view(sentences.size(0), -1, self.word_embedding_dim)

        ################################################
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        # ! although batch_first, hidden size will always be (#layes, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_input.size(0), self.lstm_dim)
        c0 = torch.zeros(self.num_layers, batch_input.size(0), self.lstm_dim)

        # ! pack the padded input
        batch_input = pack_padded_sequence(batch_input, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))
        # output shape (batch, max_length, lstm_hidden_dims)
        ################################################

        # pad output again into tensor
        output, lengths = pad_packed_sequence(output, batch_first=True) # (batch, max_seq_length, hidden_size)
        mask = torch.arange(output.size(1)).expand(len(lengths), -1) < lengths.unsqueeze(1)

        output = self.relu(self.fc(output)) # (batch, max_seq_length, vocab_size)

        # deal with padded entries
        output = torch.where(mask.unsqueeze(-1), output, torch.tensor(float('-inf')))

        output = output.view(-1, self.vocab_length)

        output = self.softmax(output)
        
        return output # shape (batch* max_seq_length, vocab_size)

    def predict(self, words):
        """ predict the rest of the sentence given words which elongates
        :param words: ndarray with shape (seq_length, )
        """
        self.eval()
        # add batch dimension
        input = torch.tensor(words[None, :], dtype=torch.long)
        input = self.word_embedding_lookup(input).view(1, -1, self.word_embedding_dim)
        h0 = torch.zeros(self.num_layers, 1, self.lstm_dim)
        c0 = torch.zeros(self.num_layers, 1, self.lstm_dim)
        output, _ = self.rnn_lstm(input, (h0, c0))
        out = output.contiguous().view(-1, self.lstm_dim) # (seq_length, lstm_dims)
        out = self.relu(self.fc(out)) # (seq_length, output_dims)
        out = self.softmax(out) # (seq_length, vocab_size)
        # return prediction of the next vocab, shape (1, vocab_size)
        return out[-1:]

        
