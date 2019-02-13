from mnist import MNIST
from myNN import MYnn
import numpy as np
import sys
import copy
import argparse
import cv2

if __name__ == '__main__':
    ## parse the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to dataset for training, path to the image for testing')
    # parser.add_argument('-nlayer', action='store', dest='num_layer', help='number of hidden layers')
    parser.add_argument('-t', '--test', action='store_true', help='use test mode, load network weights from files', default=False)
    # parser.add_argument('-p', '--path_weight')    TO DO !
    parser.add_argument('-i', '--initialize', action='store_false', help='initialize the weight with uniform distribution, it is gaussian with fan-in by default', default=True)
    parser.add_argument('-n', '--nhidden', type=int, action='append', help='number of hidden neurons, default=64', default=[64])
    parser.add_argument('-l', '--nlayer', type=int, help='number of hidden layers, default=1', default=1)
    parser.add_argument('-b', '--batch_size', type=int, help='size of the mini-batch in SGD, defalt=128', default=128)
    parser.add_argument('-r', '--learn_rate', type=float, help='learning rate, default=1e-5', default=1e-5)
    parser.add_argument('-a', '--annealing', action='store_false', help='do not apply annealing, which is applied by default', default=True)
    parser.add_argument('-sh', '--shuffle', action='store_false', help='do not apply shuffle, which is applied in each epoch by default', default=True)
    parser.add_argument('-s', '--sigmoid', choices=['lr', 'ht', 'relu'], help='nonlinear activations, default=logistic', default='lr')
    parser.add_argument('-la', '--lamb', type=float, help='regularization paremeters, default=0', default=0.)
    parser.add_argument('-m', '--momentum', action='store_false', help='do not apply momentum, which is applied by default', default=True)
    parser.add_argument('-e', '--max_epoch', type=int, help='maximum epochs the training can run, default=5000', default=5000)
    parser.add_argument('-d', '--debug', action='store_true', help='enable debuge mode, check numerical gradients', default=False)
    parser.add_argument('-sv', '--save_weights', action='store_true', help='store the weights between each layers, which is not saved by default', default=False)
    args = parser.parse_args()

    print('\n------------------------------')
    print('Parameters:')
    print('- path to dataset/test_image: {}'.format(args.data_path))
    if args.test:
        print('TEST mode on:')
        print('- number of hidden neurons = {}'.format(args.nhidden))
        print('- number of hidden layers = {}'.format(args.nlayer))
    else: 
        print('TRAINING mode on:')
        if args.initialize:
            print('- initialize using Gaussing with fan-in')
        else:
            print('- initialize with uniform distribution')
        print('- number of hidden neurons = {}'.format(args.nhidden))
        print('- number of hidden layers = {}'.format(args.nlayer))
        print('- batch size = {}'.format(args.batch_size))
        print('- learning rate = {}'.format(args.learn_rate))
        print('- annealing = {}'.format(args.annealing))
        print('- shuffle = {}'.format(args.shuffle))
        print('- sigmoid = {}'.format(args.sigmoid))
        print('- lambda = {}'.format(args.lamb))
        print('- momentum = {}'.format(args.momentum))
        print('- max epoch = {}'.format(args.max_epoch))
        print('- debug: check numerical gradients = {}'.format(args.debug))
        print('- save weights = {}'.format(args.save_weights))
    print('------------------------------\n')

    if args.test:
        # load test image
        image = cv2.imread(args.data_path, 0) # load as grayscale
        image.resize((28, 28))
        image_norm = np.array(image)/127.5 - 1
        x = image_norm.reshape(28*28, 1)  # 784x1
        x = np.vstack((x, np.ones(x.shape[1])))
        # prepare the network
        mynn = MYnn(copy.deepcopy(x), np.random.uniform(-1, 1, size=(10, 1)))
        for n in args.nhidden:
            mynn.add_layer(n, args.initialize)
        # perform a test
        prob, idx_onehot = mynn.test(x)
        idx = np.where(idx_onehot==1.)
        print("The number is {} with probability {}".format(idx[0][0], prob[0]))
    else:
        ## load data
        print(' Loading MNIST dataset...')
        mndata = MNIST(args.data_path)
        images, labels = mndata.load_training()
        images_ts, labels_ts = mndata.load_testing()
        # # use partial dataset
        # images, labels = images[:20000], labels[:20000]
        # images_ts, labels_ts = images_ts[-2000:], labels_ts[-2000:]

        ## rearrange the inputs data (normalization)
        print(' Preparing MNIST data...')
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
        mynn.fit(args.batch_size, args.learn_rate, args.annealing, args.shuffle, args.sigmoid, args.lamb, args.momentum, args.max_epoch, args.save_weights)
        mynn.plot_acc()
        mynn.plot_E()
        print('Final accuracy on train, hold-out, and test are {}, {}, {} respectively').format(mynn.acc_hist[0][-3], mynn.acc_hist[1][-3], mynn.acc_hist[2][-3])

