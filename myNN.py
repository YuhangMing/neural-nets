import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import copy
# import math

class MYnn(object):
    def __init__(self, x, t, xho=None, tho=None, xts=None, tts=None):
        self.x = np.vstack((x, np.ones(x.shape[1])))  # training data
        self.t = t              # training label
        self.xho = np.vstack((xho, np.ones(xho.shape[1])))   # hold-out data
        self.tho = tho          # hold-out label
        self.xts = np.vstack((xts, np.ones(xts.shape[1])))   # test data
        self.tts = tts          # test label
        self.sigmoid = 'lr'     # sigmoid, default as logistic
        self.Wmat = []          # weight matrices, input -> output
        # initialized as no hidden layer nn
        self.Wmat.append(np.random.uniform(-1, 1, size=(self.t.shape[0], self.x.shape[0])))
        # error and accuracy list, [tn, ho, ts]
        self.err_hist = [[], [], []]
        self.acc_hist = [[], [], []]
        
    def add_layer(self, num_hid=64, initial=False):
        tmpWmat = self.Wmat.pop()
        d_next, d_pre = tmpWmat.shape
        # Check initial method
        if initial:
            s = np.power(d_pre, -0.5)
            Wpre = np.random.normal(0, s, size=(num_hid, d_pre))
            s = np.power(num_hid+1, -0.5)
            Wnext = np.random.normal(0, s, size=(d_next, num_hid+1))
        else:
            Wpre = np.random.uniform(-1, 1, size=(num_hid, d_pre))
            Wnext = np.random.uniform(-1, 1, size=(d_next, num_hid+1))
        # add weighte matrix into the list
        self.Wmat.append(Wpre)
        self.Wmat.append(Wnext)
    
    ############################# DEBUG ##############################################    
    def check_gradient(self, num=2, epsilon=1e-2):
        # loop through all samples
        for ind in np.random.randint(0, self.x.shape[1], size=num):
            print('Sample #{}:'.format(ind))
            sample = self.x[:,ind].reshape((self.x.shape[0], 1))
            label = self.t[:,ind].reshape((self.t.shape[0], 1))
            # negative gradient from backprop
            grad = self.backprop(sample, label, copy.deepcopy(self.Wmat))
            # negative gradient by numerical approximation
            approx = self.numerical_approx(sample, label, epsilon)
            # find difference
            for g, a in zip(grad, approx):
                print('difference: max={}, min={}'.format(np.max(g-a),np.min(g-a)))
    def numerical_approx(self, X, T, epsilon):
        approx = []
        for idx, W in enumerate(self.Wmat):
            tmpApp = np.zeros(W.shape)
            for j in xrange(W.shape[0]):
                for i in xrange(W.shape[1]):
                    tmpW = copy.deepcopy(W)
                    tmpWmat = copy.deepcopy(self.Wmat)
                    tmpW[j, i] = tmpW[j, i]+epsilon
                    tmpWmat[idx] = tmpW
                    y1,_ = self.forward(X, tmpWmat)
                    tmpW = copy.deepcopy(W)
                    tmpWmat = copy.deepcopy(self.Wmat)
                    tmpW[j, i] = tmpW[j, i]-epsilon
                    tmpWmat[idx] = tmpW
                    y2,_ = self.forward(X, tmpWmat)
                    tmpApp[j, i] = -(self.error(y1, T) - self.error(y2, T))/(2*epsilon)
            approx.append(tmpApp)
        return approx
    #####################################################################################
    
    def forward(self, X, Wmat):
        ################################
        # x is num_input_unit x N
        # z is [x, h1, h2, ...]
        # y is num_output_unit x N
        ################################
        # loop through all weight matrix, use sigmoid except for the last one
        Z = [X]
        for idx in xrange(len(Wmat)):
            if idx < len(Wmat)-1:
                # hidden layer
                ah = Wmat[idx].dot(Z[idx])
                tmpZ = np.vstack((self.activation(ah), np.ones(ah.shape[1]))) # add a bias term to the end
                Z.append(tmpZ)
            else:
                # output layer
                ao = Wmat[idx].dot(Z[-1])
                Y = self.softmax(ao)
        return Y, Z
    
    def backprop(self, X, T, Wmat):
        ############################################
        # X: batch of training data
        # T: target labels
        # Return a list of gradients, input -> output
        ############################################
        # forward
        y, z = self.forward(X, Wmat)
        # calculate gradients
        gradients = []
        # output layer
        delta_next = T - y
        grad = delta_next.dot(z[-1].T)
        gradients.insert(0, grad)
        # calculate gradients - hidden, loop through all hidden layers
        for idx in xrange(-1, -1*len(z), -1):
            g_prime = self.deriv(z[idx][:-1, :])
            delta_next = g_prime * (Wmat[idx][:, :-1].T.dot(delta_next))
            grad = delta_next.dot(z[idx-1].T)
            gradients.insert(0, grad)  
        return gradients
    
    def softmax(self, a):
        numer = np.exp(a)
        if a.ndim > 1:
            output = numer / np.sum(numer, axis=0)[None, :]
        else:
            output = numer / np.sum(numer)
        return output
    
    def activation(self, a):
        if self.sigmoid == 'lr':
            return 1/(1+np.exp(-a))
        elif self.sigmoid == 'ht':
            return 1.7159*np.tanh(a*2/3)
        elif self.sigmoid == 'relu':
            return a*(a>0)
        else:
            print('sigmoid not supported')
            sys.exit(0)
    
    def deriv(self, z):
        # check which sigmoid to use
        if self.sigmoid == 'lr':
            return z*(1-z)
        elif self.sigmoid == 'ht':
            return (1.7159*2/3)*(1-np.power((z/1.7159), 2))
        elif self.sigmoid == 'relu':
            return 1.*(z>0)
        else:
            print('sigmoid not supported')
            sys.exit(0)
        return par_z_ah
    
    def fit(self, batch_size=128, eta=1e-5, annealing = True, shuffle=False, sigmoid='lr', lamb=0., momentum=False, max_epoch=5000):
        # set sigmoid
        self.sigmoid = sigmoid
        # check momentum
        dWmat = np.zeros(len(self.Wmat)).tolist()
        alpha = 0.9
        # set epoch parameters
        early_flag = 0
        epoch = 0
        # accuracy & loss
        y, _ = self.forward(self.xho, copy.deepcopy(self.Wmat))
        errho = self.error(y, self.tho)
        self.append(copy.deepcopy(self.Wmat))
        # start training
        print('####### START TRAINING #######')
        while True:
            # shuffle data
            if shuffle:
                np.random.seed(253)
                np.random.shuffle(np.transpose(self.x))
                np.random.seed(253)
                np.random.shuffle(np.transpose(self.t))
            # annealing learning rate
            eta_epoch = eta/(1 + epoch/10000) if annealing else eta
            # loop through all patterns
            Wmat_new = copy.deepcopy(self.Wmat)
            for idx in xrange(0, self.x.shape[1], batch_size):
                # find the mini-batch
                end_idx = idx + batch_size
                if end_idx > self.x.shape[1]:
                    end_idx = self.x.shape[1]
                sample = self.x[:, idx:end_idx].reshape((self.x.shape[0], end_idx-idx))
                label = self.t[:, idx:end_idx].reshape((self.t.shape[0], end_idx-idx))
                # back propagation on mini-batch
                gradients = self.backprop(sample, label, Wmat_new)
                # update weight matrix
                for idx in xrange(len(gradients)):
                    if momentum:
                        dWmat[idx] = alpha*dWmat[idx] + eta_epoch*gradients[idx]/batch_size
                    else:
                        dWmat[idx] = eta_epoch*gradients[idx]/batch_size
                    Wmat_new[idx] = Wmat_new[idx] + dWmat[idx] - 2*lamb*Wmat_new[idx]
            # check error on holdout set
            y, _ = self.forward(self.xho, Wmat_new)
            errho_new = self.error(y, self.tho)
            # count epoch
            epoch = epoch + 1
            if epoch % 100 == 0:
                print(epoch)
                print('{} - {}'.format(errho, errho_new))
                print('step {}'.format(eta_epoch))
            # check for updates
            if errho_new < errho:
                # update weight vector
                self.Wmat, errho = copy.deepcopy(Wmat_new), copy.deepcopy(errho_new)
                self.append(copy.deepcopy(self.Wmat))
                # reset early stop flag
                early_flag = 0
            else:
                early_flag = early_flag + 1
                # stop training, successive 3 times
                if early_flag < 3:
                    # update and continue
                    self.Wmat, errho = copy.deepcopy(Wmat_new), copy.deepcopy(errho_new)
                    # add accuracy and error term
                    self.append(copy.deepcopy(self.Wmat))
                else:
                    print('#### Converged after {} epochs ####'.format(epoch))
                    break
            # check maximum epoch number
            if epoch >= max_epoch:
                print('#### Maximum training epochs reached ####')
                break 
    
    def error(self, Y, T):
        # compute cross-entropy error on output and target
        E = -1 * np.sum(T*np.log(Y))
        # return normalized loss
        return E/Y.shape[1]
    
    def accuracy(self, y, t):
        max_val = np.max(y, axis=0)
        y_onehot = np.equal(y, max_val[None, :]).astype(float)
        comp = np.sum(np.equal(t, y_onehot).astype(float), axis=0)
        acc = np.equal(comp, 10.).astype(float)
        return np.sum(acc)/y.shape[1]
    
    def append(self, Wmat):
        # tn
        y, _ = self.forward(self.x, Wmat)
        self.acc_hist[0].append(self.accuracy(y, self.t))
        self.err_hist[0].append(self.error(y, self.t))
        # ho
        y, _ = self.forward(self.xho, Wmat)
        self.acc_hist[1].append(self.accuracy(y, self.tho))
        self.err_hist[1].append(self.error(y, self.tho))
        # ts
        y, _ = self.forward(self.xts, Wmat)
        self.acc_hist[2].append(self.accuracy(y, self.tts))
        self.err_hist[2].append(self.error(y, self.tts))  
    
    def plot_acc(self):
        x = np.arange(len(self.acc_hist[0]))
        plt.plot(x, np.asarray(self.acc_hist[0]))
        plt.plot(x, np.asarray(self.acc_hist[1]))
        plt.plot(x, np.asarray(self.acc_hist[2]))
        plt.legend(['train', 'hold-out', 'test'], loc='lower right')
        plt.title('accuracy vs epoch')
        plt.show()
    
    def plot_E(self):
        x = np.arange(len(self.err_hist[0]))
        plt.plot(x, np.asarray(self.err_hist[0]))
        plt.plot(x, np.asarray(self.err_hist[1]))
        plt.plot(x, np.asarray(self.err_hist[2]))
        plt.legend(['train', 'hold-out', 'test'], loc='upper right')
        plt.title('loss function vs epoch')
        plt.show()
