
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from myRNN import myRNN

def split_data(seq, bsz):
    # split the data into truncks of batches
    batch_len = int(np.floor(len(seq)/bsz))
#     print(batch_len)
    batch_ind = np.arange(0, len(seq), batch_len)
    split_seq = []
    for i in range(bsz):
        split_seq.append(seq[i*batch_len:(i+1)*batch_len])
    return split_seq, batch_ind

def get_batch(bseq, seq_len, bsz, to_idx, start_index):
    # (seq_length, batch_size, feature_size)
    tmpd, tmpt = [], []
    for i in range(bsz):
        tmpseq = bseq[i]
        
        data_seq = tmpseq[start_index:start_index+seq_len]
        targ_seq = tmpseq[start_index+1:start_index+1+seq_len]
        
        tmpd.append([to_idx[w] for w in data_seq])
        tmpt.append([to_idx[w] for w in targ_seq])
    data = autograd.Variable(torch.t(torch.LongTensor(tmpd)).cuda())
    target = autograd.Variable(torch.t(torch.LongTensor(tmpt)).cuda())
    return data, target

def evaluation(data, char2idx, ind, seq_length):
    running = 0
    for i in range(len(ind)):
        sentence_in, targets = get_batch(data, seq_length, batch_size, char2idx, ind[i])
        output = model(sentence_in)
        loss = loss_function(output.view(-1, 93), targets.view(-1))
        running += loss.data.cpu().numpy()[0]
    loss = running/len(ind)
    return loss

def plot(train, hold):
    x = np.arange(len(train))
    plt.plot(train)
    plt.plot(hold)
    plt.legend(['train', 'hold-out'])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(1)
    if not torch.cuda.is_available():
        raise('GET A GPU BEFORE RUNNING')

    ## parse the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path of the input data')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size in GPU learning, defalt=100', default=100)
    parser.add_argument('-s', '--seq_length', type=int, help='length of input sequence, default=25', default=25)
    parser.add_argument('-e', '--nembed', type=int, help='dimension of word embedding, default=64', default=64)
    parser.add_argument('-n', '--nhidden', type=int, help='number of hidden neurons, default=100', default=100)
    parser.add_argument('-d', '--dropout', type=float, help='dropout rate, default=0.5', default=0.5)
    parser.add_argument('-r', '--learn_rate', type=float, help='learning rate, default=1e-5', default=0.01)
    args = parser.parse_args()

    ## set parameter
    batch_size = args.batch_size
    seq_length = args.seq_length
    EMBEDDING_DIM = args.nembed
    HIDDEN_DIM = args.nhidden

    ## prepare the data
    with open(pars.data_path+'input.txt', 'r') as f:
        data = f.read()
    print('Length of the original data is {}'.format(len(data)))
    vocab = list(set(data))
    vocab_size = len(vocab)
    print('Vocabulary size is: {}'.format(vocab_size))
    # save pair dictionaries
    char2idx = dict((char, idx) for idx, char in enumerate(vocab))
    idx2char = dict((idx, char) for idx, char in enumerate(vocab))
    np.save('./char2idx.npy', char2idx)
    np.save('./idx2char.npy', idx2char)

    ## split dataset
    train = data[:int(len(data)*8/10)]
    holdout = data[int(len(data)*8/10):]
    print('Length of train and hold-out data are {} and {}'.format(len(train), len(holdout)))
    data_btn, ind_btn = split_data(train, batch_size)
    data_bho, ind_bho = split_data(holdout, batch_size)

    ## setup the model
    model = myRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, dropout=args.dropout, batch=batch_size)
    model = model.cuda()
    print(model)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learn_rate)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training
    ind_tn = np.arange(0, len(data_btn[0])-seq_length+1, seq_length)
    ind_ho = np.arange(0, len(data_bho[0])-seq_length+1, seq_length)
    loss_tn = []
    loss_ho = []
    loss_rn = []
    pre_loss = np.inf
    stop_flag = 0
    since = time.time()
    # epoch = 0
    # while True:  # again, normally you would NOT do 300 epochs, it is toy data
    #     epoch += 1
    for epoch in range(700):
    #     np.random.shuffle(indexs)
        print('-'*30)
        print('epoch {}:\nTraining...'.format(epoch))
        running_tn = 0
        # reset memory in new epoch
        model.init_hidden(batch_size)
        model.train()
        for i in range(len(ind_tn)):
            ## Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # detaching it from its history on the last instance.
    #         model.hidden = model.init_hidden()
            model.hidden = tuple(autograd.Variable(v.data).cuda() for v in model.hidden)
            ## Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in, targets = get_batch(data_btn, seq_length, batch_size, char2idx, ind_tn[i])
            ## Step 3. Run our forward pass.
            output = model(sentence_in)
            ## Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(output.view(-1, vocab_size), targets.view(-1))
            running_tn += loss.data.cpu().numpy()[0]
            loss.backward()
            optimizer.step()
            print('{0:.2f}%'.format((i+1)*100/len(ind_tn)), end='\r')
        loss_rn.append(running_tn/len(ind_tn))
        print('\nValidating...')
        # check validation set
        model.eval()
        loss = evaluation(data_bho, char2idx, ind_ho, seq_length)
        # check convergence criteria
    #     print('val loss pre - cur: {0:.6f} - {1:.6f}'.format(pre_loss, loss))
        if loss < pre_loss:
            torch.save(model.state_dict(), './model.pth')
            loss_ho.append(loss)
            loss_tn.append(evaluation(data_btn, char2idx, ind_tn, seq_length))
            pre_loss = loss
            stop_flag = 0
        else:
            # stop training if loss on val increase for 3 successive epochs
            if stop_flag < 2:
                torch.save(model.state_dict(), './model.pth')
                loss_ho.append(loss)
                loss_tn.append(evaluation(data_btn, char2idx, ind_tn, seq_length))
                pre_loss = loss
                stop_flag += 1
            else:
                break
    #     print('Stop flag is: {}'.format(stop_flag))
        print('Running - Train - Holdout: {0:.6f} - {1:.6f} - {2:.6f}'.format(loss_rn[-1], loss_tn[-1], loss_ho[-1]))
        print('time consuming: {0:.1f}s'.format(time.time()-since))
        since = time.time()
    #     print('loss at epoch {} = {}'.format(epoch, loss.data))

    # plot loss vs epoch
    plot(loss_tn, loss_ho)

    # load model if restart kernel
    model = myRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, dropout=0.3)
    model.load_state_dict(torch.load('./model.pth'))
    model = model.cuda()
    model.init_hidden()
    model.eval()
    char2idx = np.load('./char2idx.npy').item()
    idx2char = np.load('./idx2char.npy').item()

    ## Generating:##
    music = '<start>'
    print(music, end='')
    while True:
        if '<end>' in music:
            break
        else:
            if len(music) < seq_length:
                seq_in = music
            else:
                seq_in = music[-seq_length:]
            idx_in = autograd.Variable(torch.LongTensor([char2idx[w] for w in seq_in]).cuda())
            char = model.predict(idx_in, idx2char, 2)
            music += char
            print(char, end='')
    # write to file
    with open('output.txt', 'w') as f:
        output_music = f.write(music)



