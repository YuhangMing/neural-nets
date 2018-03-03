
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class myRNN(nn.Module):
    def __init__(self, nvocab, nemb, nhid, nlayers=1, dropout=0.5, batch=1):
        super().__init__()
        self.encoder = nn.Embedding(nvocab, nemb)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(nemb, nhid, dropout=dropout)
        # The linear layer that maps from hidden state space to tag space
        self.decoder = nn.Linear(nhid, nvocab)
        self.nhid = nhid
        self.nlayers = nlayers
        self.bsz = batch
        self.hidden = self.init_hidden(batch)

    def init_hidden(self, bsz=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         if self.rnn_type == 'LSTM':
        return (autograd.Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda()),
                autograd.Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda()))
#         else:
#             return autograd.Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda())
    
    def forward(self, sentence):
        embeds = self.encoder(sentence)
        # input of lstm should be (seq_length, batch_size, feature_size)
        # using view to ensure the input size
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), self.bsz, -1), self.hidden)
        # lstm_out.size = lstm_in.size 
        # (nlayer*bsz*nhidden, nlayer*bsz*nhidden) = (hid_state, cell_state)
#         output = self.decoder(lstm_out.view(len(sentence), -1))
        output = self.decoder(lstm_out)
#         print(lstm_out) (25x32x100)
#         print(output) (25x32x93)
        return output

    def predict(self, sentence, idx2char, T=1):
        embeds = self.encoder(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        output = self.decoder(lstm_out.view(len(sentence), -1))
        # convert to char
        pred = output.data.cpu().numpy()[-1, :]
        smT = np.exp(pred/T) / np.sum(np.exp(pred/T))
        # random sample from the predicted distribution
        ind = np.random.choice(len(smT), p=smT)
        char = idx2char[ind]
        return char
