from mnist import MNIST
from myNN import MYnn
import numpy as np
import sys
import copy
import argparse

if __name__ == '__main__':
    ## parse the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path of the dataset')
    # parser.add_argument('-nlayer', action='store', dest='num_layer', help='number of hidden layers')
    parser.add_argument('-i', '--initialize', action='store_false', help='initialize the weight with uniform distribution, it is gaussian with fan-in by default')
    parser.add_argument('-n', '--nhidden', type=int, action='append', help='number of hidden neurons, default=64', default=[64])
    parser.add_argument('-l', '--nlayer', type=int, help='number of hidden layers, default=1', default=1)
    parser.add_argument('-b', '--batch_size', type=int, help='size of the mini-batch in SGD, defalt=128', default=128)
    parser.add_argument('-r', '--learn_rate', type=float, help='learning rate, default=1e-5', default=1e-5)
    parser.add_argument('-a', '--annealing', action='store_false', help='do not apply annealing, which is applied by default')
    parser.add_argument('-sh', '--shuffle', action='store_false', help='do not apply shuffle, which is applied in each epoch by default')
    parser.add_argument('-s', '--sigmoid', choices=['lr', 'ht', 'relu'], help='nonlinear activations, default=logistic', default='lr')
    parser.add_argument('-la', '--lamb', type=float, help='regularization paremeters, default=0', default=0.)
    parser.add_argument('-m', '--momentum', action='store_false', help='do not apply momentum, which is applied by default')
    parser.add_argument('-e', '--max_epoch', type=int, help='maximum epochs the training can run, default=5000', default=5000)
    parser.add_argument('-d', '--debug', action='store_true', help='enable debuge mode, check numerical gradients')
    args = parser.parse_args()

    ## load data
    mndata = MNIST(args.data_path)
    images, labels = mndata.load_training()
    images_ts, labels_ts = mndata.load_testing()
    # # use partial dataset
    # images, labels = images[:20000], labels[:20000]
    # images_ts, labels_ts = images_ts[-2000:], labels_ts[-2000:]

    ## rearrange the inputs data (normalization)
    images_norm = [np.array(i)/127.5 - 1 for i in images]
    # 50000 train, 10000 hold out
    x_tn = np.asarray(images_norm[:50000]).T    # 784x50000
    x_ho = np.asarray(images_norm[-10000:]).T    # 784x10000
    x_ts = np.asarray([np.array(i)/127.5 - 1 for i in images_ts]).T    # 784x10000

    ## create one-hot encoded target
    target = []
    for l in labels:
        t_tmp = np.zeros(10)
        t_tmp[l] = 1
        target.append(t_tmp)
    t_tn = np.asarray(target[:50000]).T    # 10x50000
    t_ho = np.asarray(target[-10000:]).T    # 10x10000
    t_ts = []
    for l in labels_ts:
        t_tmp = np.zeros(10)
        t_tmp[l] = 1
        t_ts.append(t_tmp)
    t_ts = np.asarray(t_ts).T    # 10x10000

    ## train the network
    mynn = MYnn(copy.deepcopy(x_tn), copy.deepcopy(t_tn), copy.deepcopy(x_ho),
                copy.deepcopy(t_ho), copy.deepcopy(x_ts), copy.deepcopy(t_ts))
    if args.nlayer != len(args.nhidden):
        raise('number of layers mismatch')
    for n in args.nhidden:
        mynn.add_layer(n, args.initialize)
    if args.debug:
        print('CHECKING NUMERICAL GRADIENTS')
        mynn.check_gradient()
    mynn.fit(args.batch_size, args.learn_rate, args.annealing, args.shuffle, args.sigmoid, args.lamb, args.momentum, args.max_epoch)
    mynn.plot_acc()
    mynn.plot_E()
    print('Final accuracy on train, hold-out, and test are {}, {}, {} respectively').format(mynn.acc_hist[0][-3], mynn.acc_hist[1][-3], mynn.acc_hist[2][-3])

