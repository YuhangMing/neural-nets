
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, sampler
from caltech256 import Caltech256
from random import shuffle


# In[2]:


# load & preprocess the data
normalization = transforms.Compose(
            [
                transforms.Scale((224,224)),
                transforms.RandomHorizontalFlip(),
                # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )
datasets = {
    'train': Caltech256("/datasets/Caltech256/256_ObjectCategories/", normalization, train=True),
    'hold': Caltech256("/datasets/Caltech256/256_ObjectCategories/", normalization, train=False, test=False),
    'test': Caltech256("/datasets/Caltech256/256_ObjectCategories/", normalization, train=False, test=True)
}


# In[3]:


# get dataloader
dataloaders = {
    'train': DataLoader(
            dataset = datasets['train'],
#             sampler = sampler.SubsetRandomSampler(idx_tn),
            batch_size = 32,
            shuffle = True,
            num_workers = 4
        ),
    'hold': DataLoader(
            dataset = datasets['hold'],
#             sampler = sampler.SubsetRandomSampler(idx_ho),
            batch_size = 32,
            num_workers = 4
        ),
    'test': DataLoader(
            dataset = datasets['test'],
            batch_size = 32,
            num_workers = 4
        )
}
datasizes = {
    'train': len(datasets['train']),
    'hold': len(datasets['hold']),
    'test': len(datasets['test'])
}


# In[4]:


def train_model(model, loaders, sizes, criterion, optimizer, scheduler, max_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    hist_loss = {x: [] for x in ['train', 'hold', 'test']}
    hist_acc = {x: [] for x in ['train', 'hold', 'test']}
    ofit_counter = 0
    # loop through epochs
    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch+1, max_epochs))
        print('-' * 14)
        # in each epoch, train and evaluate on hold & test
        for phase in ['train', 'hold', 'test']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()
            running_loss = 0.
            running_corrects = 0
            # iterate over data
            for data in loaders[phase]:
                # get the inputs
                inputs, labels = data
                # re-arrange labels to 0 - C-1
                labels -= 1
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.long()[:,0].cuda())
                else:
                    raise('Get a GPU!')
                # Set gradient to zero to delete history of computations in previous epoch. 
                # Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            hist_loss[phase].append(running_loss / sizes[phase])
            hist_acc[phase].append(running_corrects / sizes[phase])
            print('{} Loss: {:.7f} Acc: {:.4f}'.format(phase, hist_loss[phase][-1], hist_acc[phase][-1]))  
        print()
        if hist_loss['hold'][-1] < best_loss:
            best_loss = hist_loss['hold'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            ofit_counter = 0
        else:
            ofit_counter += 1
            # stop training if the loss on hold-out set increases for 2 successive epochs
            if ofit_counter < 2:
                best_loss = hist_loss['hold'][-1]
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best hold-out loss: {:.4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, hist_loss, hist_acc


# In[5]:


# load pretrained vgg 16
model_vgg = models.vgg16(pretrained=True)
# model_vgg = models.vgg16_bn(pretrained=True)


# In[6]:


# set training parameters
for param in model_vgg.parameters():
    param.requires_grad = False
model_vgg.classifier._modules['6'] = nn.Linear(4096, 256)
model_vgg = model_vgg.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_vgg = optim.SGD(filter(lambda p: p.requires_grad, model_vgg.parameters()), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_vgg, step_size=20, gamma=0.1)


# In[7]:


## train the new FC layer
model_vgg, loss, acc = train_model(model_vgg, dataloaders, datasizes, criterion,
                                   optimizer_vgg, exp_lr_scheduler, max_epochs=50)


# In[8]:


# plot loss and accuracy over epochs
def plot(y, criteria, location):
    x = range(len(y['train']))
    for phase in ['train', 'hold', 'test']:
        plt.plot(x, np.asarray(y[phase]))
    plt.legend(['train', 'hold-out', 'test'], loc=location)
    plt.title(criteria +' vs. epoch')
    plt.grid()
    plt.show()

plot(loss, 'Loss', 'upper right')
print('Final loss for training, hold-out, test sets are {:.4f}, {:.4f}, {:.4f} respectively.'.format(loss['train'][-1], loss['hold'][-1], loss['test'][-1]))
print()
plot(acc, 'Accuracy', 'lower right')
print('Final accuracy for training, hold-out, test sets are {:.4f}, {:.4f}, {:.4f} respectively.'.format(acc['train'][-1], acc['hold'][-1], acc['test'][-1]))


# In[9]:


## Visualize the convolution filter
idx = np.random.randint(datasizes['train'])
sample = datasets['train'][idx][0].view(1,3,224,224)

# get convolution layers
# First
conv1 = list(model_vgg.features.children())[0].cuda()
output1 = conv1(Variable(sample.cuda()))  # variable of size 1x64x224x224
# Last
conv2 = nn.Sequential(*list(model_vgg.features.children())[:-2]).cuda()
output2 = conv2(Variable(sample.cuda()))


# In[10]:


# display input image
in1 = sample.view(3, 224, 224).numpy().transpose((1,2,0))
in1 = in1 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
plt.imshow(np.clip(in1, 0, 1))
plt.title('Input image')
plt.show()

# randomly choose 8 filters to display
def disp_filter(output, channel, size, layer):
    out = output.data.view(channel, size, size).cpu().numpy().transpose((1,2,0))
    disp_idx = np.random.choice(range(channel), size=8)
    disp_idx.sort()
    for i, idx in enumerate(disp_idx):
        plt.subplot(2,4,i+1)
        plt.imshow(out[:,:,idx], cmap='hot')
        plt.title('channle {}'.format(idx+1))
    plt.suptitle('Examples of output images from '+layer+' conv layer')
    plt.show()

disp_filter(output1, 64, 224, 'First')
disp_filter(output2, 512, 14, 'Last')


# In[11]:


## Visualize the weight of first conv layer
weights = conv1.weight.data.cpu().numpy().transpose((2,3,1,0))
plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    wts = weights[:,:,:,i]
    wts -= np.min(wts)
    wts /= np.max(wts)
    plt.imshow(wts)
#     plt.title('channle {}'.format(i+1))
plt.suptitle('Examples of weights from First conv layer')
plt.show()

