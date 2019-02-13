from myNN import MYnn
import numpy as np
import copy
import cv2

def initialize_nn():
    # prepare the network
        mynn = MYnn(np.random.uniform(-1, 1, size=(28*28, 1)), np.random.uniform(-1, 1, size=(10, 1)))
        for n in [64]:
            mynn.add_layer(n, True)

def detect(image):
    if np.sum(image[:]) == 28*28*255:
        num = 0
        # print("Empty block, set to 0")
    else:
        image_norm = np.array(image)/127.5 - 1
        x = image_norm.reshape(28*28, 1)  # 784x1
        x = np.vstack((x, np.ones(x.shape[1]))) #785x1
        # prepare the network
        mynn = MYnn(np.random.uniform(-1, 1, size=(28*28, 1)), np.random.uniform(-1, 1, size=(10, 1)))
        for n in [64]:
            mynn.add_layer(n, True)
        # perform a test
        prob, idx_onehot = mynn.test(x)
        # print(prob)
        # print(idx_onehot)
        idx = np.where(idx_onehot==1.)
        # print("The number is {} with probability {}".format(idx[0][0], prob[0]))
        num = idx[0][0]

    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    return num