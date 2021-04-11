import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import asgn_1 as SANKETNET
import torchvision.transforms.transforms as transforms
import pickle
import torchvision.models as models
import torchvision.datasets as datasets
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import PIL.Image as pil
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Action
import box
import time

Lambda = transforms.Lambda

from torch.utils.tensorboard import SummaryWriter

# ===== Problem 1.1 ========

# choose pretrained model of choice
# get from torch vision, ie resnet, no need to train, should be already trained with weights
# should instantiate without need to load dictionary?

# do a forward call,
# push peppers.jpg through it, output top 3 cats, after preprocessigng (check model docs)
# print output, basically

# Visualize the feature maps at different layers as images (may have to normalizing between 0 and
# 1 for visualizing)

# Note structure, interpretability, challenges to interpretability
# q why would it matter if i could get original subset or not for pretrained weights?
# q does size of working dataset determine which transfer method to use?....is there a vanishing
# q can you combine all the matrix transforms from pretrain model, to get the final output feature?

#  gradient implication...

# .torch/models?
# .torch/

# Data Loader for single image dataset.

### NOTE FROM DOC ###

# Code for processing data samples can get messy and hard to maintain;  we ideally want our
# dataset code to be decoupled from our model training code for better readability and
# modularity.  PyTorch provides two data primitives: torch.utils.data.DataLoader and
# torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data.
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable
# around the Dataset to enable easy access to the samples.
#
# PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
# subclass torch.utils.data.Dataset and implement functions specific to the particular  data.

# Data does not always come in its final processed form that is required for training machine
# learning algorithms. We use transforms to perform some manipulation of the data and make it
# suitable for training.
#
# All TorchVision datasets have two parameters -transform to modify the features and
# target_transform to modify the labels - that accept callables containing the transformation
# logic. The torchvision.transforms module offers several commonly-used transforms out of the
# box. for params input ToTeensor() or Lambda(lambda y:

# opens class dictionary as object, just for returning what values forward call returns



pepper_path = './resources/peppers.jpg'
classes = None
with open('./resources/imagenet_class_index.json', 'r') as j:
    classes = json.load(j)
classes = {int(i): j for i, j in classes.items()}


def transform_peppers(img_path):
    """
    """
    # does not close pil obj
    I = open(pepper_path, 'rb')
    _img = pil.open(I)
    # resize image to 224x224
    img = transforms.Resize((224, 224)).forward(_img)
    # 0 to 1 tensor
    img = transforms.ToTensor()(img)
    # standardize with mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (RGB)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(img)
    print(img.shape)
    _img.close()
    I.close()
    return img


# Implementing Custom size 1 Dataset Object from Dataset primitive. Dataloader will wrap this
# class and generate batches.
class Datum(Dataset):
    def __init__(self, img_path, transform=None, target_transform=None):
        self.img_labels = None
        self.img = img_path  # here will just be a direct path to peppers.
        self.transform = transform
        self.target_transform = target_transform

    # NOTE FROM DOC:
    # The __getitem__ function loads and returns a sample from the dataset at the  given index
    # idx. Based on the index, it identifies the image’s location on disk, converts that  to a
    # tensor using read_image, retrieves the corresponding label from the csv data in
    # self.img_labels, calls the transform functions on them (if applicable), and returns the
    # tensor image and corresponding label in a Python dict.
    def __getitem__(self, idx):
        # no index needed, works as gnerator
        label = ['peppers.jpg', 0]  # where 0 is the index key in the label dict
        if self.transform:
            # img is a c x h x w with 0-255 vals
            img = self.transform(self.img)  # ToTensor fn param
        if self.target_transform:
            label = label[1]  # useless in this example, but just for good form.
        return {'image': img, 'label': label}

    def __len__(self):
        return 1  # hardcoded for peppers


pepper_net = models.resnet18(pretrained=True,
                             progress=True)  # weights should be loaded to a 'cache'
peppers_datum = Datum(pepper_path, transform_peppers)
pepper_batch = DataLoader(peppers_datum, batch_size=1)


def prob1_1():
    # print(pepper_batch, type(pepper_batch))
    pepper_net.eval()
    for item in pepper_batch:  # just 1 lol
        X = item['image']
        Y_hat = pepper_net.forward(X)  # use built in callable resnet(X) in practice.

    # Unsure of Y_hat batch shape...
    for i in range(Y_hat.shape[0]):
        assert Y_hat.shape[-1] == 1000
        print(Y_hat)
        keys = torch.topk(Y_hat[i, ...], 3, dim=-1)  # should be a row.
        keys = [i.item() for i in keys.indices]
        print(classes[keys[0]], classes[keys[1]], classes[keys[2]])


# ===== Problem 1.2 ========

# Part 2 - Visualizing Feature Maps (5 points)
# Write code to visualize the feature maps in the network as images.  You will likely need to
# normalize the values between 0 and 1 to do this.
# Choose five interesting feature maps from early in the network,  five from the the middle of
# the network, and and five close to the end of the network. Display them to us and discuss the
# structure of the feature maps. Try to find some that are interpretable, and discuss the
# challenges  in doing so.

# It seems that feature maps are input x weight, not just weights, which is called filter Rather
# than create a subclassed network and re-write thew whole architecture, we can use pytorch hooks.
# hooks are like webhooks, that figuratively 'hook' whatever is passing by it, in this case the
# input and output of the module...the registerhook fn takes a fn paramter that takes signature
# hook(module, input, output), and the parent fn returns a handle to .remove() it. From docs:

# The input contains only the positional arguments given to the module.  Keyword arguments won’t
# be passed to the hooks and only to the forward.  The hook can modify the output. It can modify
# the input inplace but it will not have effect on  forward since this is   called after forward()
# is called.


def prob1_2():
    layers = ['layer1.1.conv2', 'layer2.1.conv2', 'layer4.1.conv2']
    ft_maps = []

    # hook class
    def fmap_hook(module, input, output):
        print('mapshape', output.shape)
        num_channels = output.shape[1]
        assert num_channels > 5, print(output.shape)
        num_channels = list(range(num_channels))
        inds = [random.choice(num_channels) for _ in range(5)]
        ft_maps_to_add = [output[0, i, ...] for i in inds]
        assert len(ft_maps_to_add[0].shape) == 2
        ft_maps_to_add = [transforms.ToPILImage()(o) for o in ft_maps_to_add]
        # shape 1,c,h,w
        # should output 5
        for i in ft_maps_to_add:
            ft_maps.append(i)
        return

    for p in pepper_net.named_modules():
        mod = p[1]
        if p[0] in layers:
            mod.register_forward_hook(fmap_hook)

    for item in pepper_batch:  # just 1 lol
        X = item['image']
        Y_hat = pepper_net(X)  # use built in callable resnet(X) in practice. Y-hat not actually
        # used, becasue hooked out.

    fig = plt.figure(dpi=300)
    fig.subplots(3, 5)
    fig.subplots_adjust(wspace=-5, hspace=0)  # some fomratting
    for ax, fm in zip(fig.axes, ft_maps):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.imshow(fm, cmap='jet')  # what to do with ouput channels...
    plt.show()


# todo use makegrid fn

### NOTE FROM DOC ####

# """
# Instancing a pre-trained model will download its weights to a cache directory. This directory
# can be set using the TORCH_MODEL_ZOO environment variable. See torch.utils.model_zoo.load_url()
# for details.
#
# Some  models use modules which have different training and evaluation behavior, such as batch
# normalization. To switch between these modes, use model.train() or model.eval() as appropriate.
# See train() or eval() for details.
#
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of
# 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The
# images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456,
# 0.406] and std = [0.229, 0.224, 0.225]. (the imagenet dist mean/std)  You can use the following
# transform to  normalize
# """

# From docs, essentially really confusing way of mapping Tesnor A to Tensor C, 'bridging over' a
# Tensor B. Fn takes the value in Tensor A as the column index, in this 2d case, and maps to
# row index of the same row index as val in Tensor A, as the value 1 in Tensor C. So,
# where vb = scalar val in B at index j in B , fn takes va = A[i, j] and maps it to C[i,B[j]] as
# # value = 1.

# target_transform = Lambda(lambda y: torch.zeros(
#     10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


# Actually obtaining mean/std from a dataset dist:
# The process for obtaining the values of mean and std is roughly equivalent to:
#
# import torch
# from torchvision import datasets, transforms as T
#
# transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# dataset = datasets.ImageNet(".", split="train", transform=transform)
#
# means = []
# stds = []
# for img in subset(dataset):
#     means.append(torch.mean(img))
#     stds.append(torch.std(img))
#
# mean = torch.mean(torch.tensor(means))
# std = torch.mean(torch.tensor(stds))

# q detach weights for problem 3

# === Problem 2 ===

# Transfer Learning with a Pre-Trained CNN (20 points) For this
# problem you must use PyTorch. We will do image classification using the Oxford Pet Dataset. The
# dataset consists of 37 categories with about 200 images in each of them. You can find the
# dataset here: http://www.robots.ox.ac.uk/~vgg/data/pets/ Rather than using the final ‘softmax’
# layer of the CNN as output to make predictions as we did in problem 1, instead we will use the
# CNN as a feature extractor to classify the Pets dataset.

# For each image, grab features from the last hidden layer of the neural network, which will be
# the layer before the 1000-dimensional output layer (around 500– 6000 dimensions).
#
# You will
# need to resize the images to a size compatible with your network (usually 224 × 224 × 3,
# but look at the documentation for the
# pre-trained system you selected). You should grab the output just after the last hidden layer
# or after global pooling (if it is 1000-dimensional, you will know you did it wrong).
#
# After you
# extract these features for all of the images in the dataset, normalize them to unit length by
# dividing by the L2 norm.
#
# Train a linear classifier of your choice1 with the training CNN
# features, and then classify the test CNN features. Report mean-per-class accuracy and discuss
# the classifier you used.

# datasets reorganized with bash script
# Unzip, Create Dataset object

# annot_table = pd.read_csv(
#     './resources/annotations/list.txt',
#     ' ',
#     skiprows=(0, 1, 2, 3, 4, 5),
#     names=["file", "class id", "species", "breed"],
#     header=0
# )
# targets = annot_table.loc[:, ['file', 'class id']]
# todo for hot helper fn, con to np array.
# this is for the fc output.....

# for first pass, torch will load pairs of path and label. it will get the path bc already
# premade dataset object....it calls it's default image loader, PIL, and then applies tranform to
# this...transform should accept image objext
# this dataset automatically determines the correct class from the directory....
# q what does scriptable mean?
### make a class dictionary out of pandas table...totally unnecessary, but pretty cool. dataset
# # object automatically creates a class dict from the directory structure....the 'classtoidx' attr
# classes = targets.copy()
# classes['file'] = classes['file'].str.extract(r'(.*?)(?:_\d)')
# classes['file'] = classes['file'].str.replace('_',' ')
# classes.groupby('class id').agg(lambda x: set(x).pop()).to_dict()['file']
###

FILE = "binary"
BOX_AUTH = "auth_code_here"

hypes = {
    "EPOCHS": 5,
    "BATCH": 1,  # 32, 16, 64
    "RATE": .01,  # .1 .05 .01 .05, .02 .025
    "MOMENTUM": .2,
    "RC": .005  # .01 .005
}

# typically want to pass in image object here....
# you can do a transforms.Compose or just custom fn...
def transform_pets(img_obj):
    img = transforms.Resize((224, 224))(img_obj)  # must curry arg
    img = transforms.ToTensor()(img)  # 0 to 1 tensor
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    # standardize with mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (RGB)
    # typically use mean/std from training data, but in this case, leave as image net, bc that's
    # where we're transfering from....

    # print(img.shape)
    return img

def save_bin(name, f_object):
    if not os.path.exists('pickled_binaries/'):
        os.makedirs('pickled_binaries/')
    if type(f_object) == torch.Tensor:
        name = f'{name}_{int(time.time())}.pt'
        torch.save(f_object, f"pickled_binaries/{name}")
    else:
        name = f'{name}_{int(time.time())}.bin'
        file = open(f"pickled_binaries/{name}", "wb")
        pickle.dump(f_object, file)
        file.close()
    # box.upload(f'{name}')


## PETnet data
def pet_dataset():
    trainingd_root = './resources/images'
    testingd_root = './resources/test_images'
    sample_pairs_training = datasets.folder.ImageFolder(trainingd_root, transform_pets)
    sample_pairs_testing = datasets.folder.ImageFolder(testingd_root, transform_pets)
    label_dict_training = sample_pairs_training.class_to_idx
    label_dict_testing = sample_pairs_testing.class_to_idx
    return DataLoader(sample_pairs_training), DataLoader(sample_pairs_testing), label_dict_training, label_dict_testing
    

# no target transform needed here, because sanket-net already has it embedded
# data loader returns input/output, though output necessary? we need the features
# todo data must be randomized for fc layer, but not first forward for feature extraction
# to avoid overwriting the resnet architecture, or even using hooks, we will just make the self.fc
# an identity module....LOL!
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def build_maps(loader, model, samples_lim=None):
    # building the feature map...training set
    feature_maps = None
    labels = None
    count = 0
    for x, y in loader:
        count += 1
        if count > 1:
            feature_map = model(x).detach().numpy()
            feature_maps = np.vstack((feature_maps, feature_map))
            labels = np.vstack((labels, y.numpy()))
        if count == 1:
            feature_maps = model(x).detach().numpy()
            assert y.shape == (1,), print(y.shape)
            labels = y.numpy()
        if samples_lim:
            if count == samples_lim:
                break
        if count % 100 == 0:
            print(f'feature mapping {count}')

    assert feature_maps.shape[-1] == 512, print(feature_maps.shape)
    assert labels.shape[-2] == count, print(labels.shape)
    return feature_maps, labels
# todo must be MEAN PER CLASS ACCURACY....
# Problem_2 Main Function
# Taken from asgn1
def problem2b():
    ##
    # Hyperparameters
    ##
    regularization_level = 2
    print(hypes)
    # hypes = {
    #     "EPOCHS": 1000,
    #     "BATCH": 256,
    #     "RATE": .01,
    #     "MOMENTUM": .2,
    #     "RC": .05
    # }
    SANKETNET.update_hypers(hypes)
    
    # building the feature map...training set
    pet_net = models.resnet18(pretrained=True, progress=True)  # weights should be loaded to a 'cache'
    pet_net.fc = Identity()
    pet_net.eval()
    training_loader, testing_loader, training_classes, testing_classes = pet_dataset()
    
    if os.path.exists('pickled_binaries/feature_map.bin'):
        inputs = pickle.load(open('pickled_binaries/feature_map.bin', "rb"))
        targets = pickle.load(open('pickled_binaries/labels.bin', "rb"))
        test_inputs = pickle.load(open('pickled_binaries/feature_map_t.bin', "rb"))
        test_targets = pickle.load(open('pickled_binaries/labels_t.bin', "rb"))
    else:    
        _inputs, _targets = build_maps(training_loader, pet_net)
        _test_inputs, _test_targets = build_maps(testing_loader, pet_net)
        _targets = SANKETNET.hot_helper(_targets.flatten())[0]
        _test_targets = SANKETNET.hot_helper(_test_targets.flatten())[0]
        inputs, targets = SANKETNET.randomize_helper(*(_inputs,_targets))
        test_inputs, test_targets = SANKETNET.randomize_helper(*(_test_inputs,_test_targets))    
        save_bin('feature_map', inputs)
        save_bin('labels', targets)
        save_bin('feature_map_t', test_inputs)
        save_bin('labels_t', test_targets)
    network = SANKETNET.Single_Layer_Network(
        inputs,
        targets,
        test_inputs,
        test_targets,
        SANKETNET.Cross_Entropy,
        SANKETNET.Softmax_Regression,
        SANKETNET.MatMul,
        n_classes=37,
        normalizing=True
    )

    network.train(regularization_level)
    training_accuracies = SANKETNET.DataSimulatorHelper.accuracy_list(
        network.training_eval_array, "Training")
    testing_accuracies = SANKETNET.DataSimulatorHelper.accuracy_list(
        network.testing_eval_array, "Testing")
    acc_matrix = network.testing_eval_array.sum(0)
    SANKETNET.Plot.confusion(acc_matrix, testing_classes)
    mean_accuracies = {}
    for label in testing_classes:
        j = testing_classes[label]
        for (i, _),_ in np.ndenumerate(acc_matrix):
            if i == j:
                correct = i
                total = acc_matrix.sum(0)[j]
                acc = correct / total
                mean_accuracies[label] = acc

    SANKETNET.Plot.curves(
        range(hypes['EPOCHS']),
        *(training_accuracies, testing_accuracies),
        ind_label='Epochs',
        dep_label="Accuracy",
        title=f"Accuracy for Test and Training Data ({FILE})"
    )
    save_bin(f'{FILE}', network)
    save_bin(f'{FILE}_accuracies', mean_accuracies)
    return network


# code up the new model...got some options:
# 1. instantiate pretrained model, grad off for all modules, replace fc module in instance with
# new module, from asgn 1 or from pytorch...
#   a. in asgn 1 case: you'll have to manually rig up the 2...could be cool though. Basically
#   create a new class that inherits from both, instantiates supers, and re-lays out the
#   architecture but using my own net for the last step,
#   b. alternatively, instantiate resnet, and replace with pytorch module. both of these however
#   would require re-forwarding with the data everytime, and is inefficient.
# 2. Instantiate pretrained resnet, and then just delete the fc layer wwith del(x),
# then essentially call it like normal, save that output, and use that output on a fc network of
# our choice...seems like generally no hiddens in fc network anyay
#   a. can use asgn1 or pytorch
# 3. alternatively you could use hooks but not necessary because last layer...
# I vote options 2

# after that, write training loop code, iterating over the batches, just be sure to have pulled
# the features in an eval mode (resnet). for fc, doesn't matter. Would be pretty sick if i used
# my own net...will try.

# apparent questions on whether output from resnet needs to be re-normalized...will be easy to
# see in any case..

# plot mean per class accuracy.

# === Problem 3.1 ===

# For this problem you must use a toolbox. Train a CNN with three hidden convolutional layers
# that use the
# Mish activation function. Use 64 11 × 11 filters for the first layer, followed by 2 × 2 max
# pooling (stride of 2). The next two convolutional layers will use 128 3 × 3 filters followed by
# the Mish activation function. Prior to the softmax layer, you should have an average pooling
# layer that pools across the preceding feature map. Do not use a pre-trained CNN.
#
# Train your
# model using all of the CIFAR-10 training data, and evaluate your trained system on the CIFAR-10
# test data.
#
# Visualize all of the 11 × 11 × 3 filters learned by the first convolutional layer as
# an RGB image array (I suggest making a large RGB image that is made up of each of the smaller
# images, so it will have 4 rows and 16 columns). This visualization of the filters should be
# similar to the ones we saw in class. Note that you will need to normalize each filter by
# contrast stretching to do this visualization, i.e., for each filter subtract the smallest value
# across RGB channels and then divide by the new largest value across RGB channels.
#
# Display the
# training loss as a function of epochs. What is the accuracy on the test data? How did you
# initialize the weights? What optimizer did you use? Discuss your architecture and
# hyper-parameters.
#
# Solution: IMAGE SHOWING THE FILTERS TRAINING LOSS AS FUNCTION OF EPOCHS TEST
# DATA ACCURACY WEIGHT INITIALIZATION INFORMATION DESCRIBE HYPER-PARAMETERS
#
# Part 2 (20 points)
# Using the same architecture as in part 1, add in batch normalization between each of the hidden
# layers. Compare the training loss with and without batch normalization as a 4 function of
# epochs. What is the final test accuracy? Visualize the filters. Solution:
#
# Part 3 (10 points)
# Can you do better with a deeper and better network architecture? Optimize your CNN’s
# architecture to improve performance. You may get significantly better results by using smaller
# filters for the first convolutional layer. Describe your model’s architecture and your design
# choices. What is your final accuracy? Note: Your model should perform better than the one in
# Part 1 and Part 2. Solution:
def tile_and_print(input, tiles_height, tiles_width, padding = 1):
  """
  expecting a 4d weight tensor. (chan_out, chan_in, h, w). permute for matplot plot.
  This function uses permute to compose the filter map....
  """
  device = input.device
  p = padding
  w = input
  co, ci, he, wi = w.shape
  if padding:
    w = torch.cat((w,torch.ones((co,ci,p, wi), device=device)),dim=-2)
    w = torch.cat((w,torch.ones((co,ci,he+p, p), device=device)),dim=-1)
    co, ci, he, wi = w.shape
  w = w.permute(1,2,3,0)
  w = w.reshape(ci,he,wi,tiles_height, tiles_width)
  w = w.permute(0,3,1,4,2)
  w = w.reshape(ci,he*tiles_height,tiles_width*wi)
  return w

def filter_visual(Model):
    W = Model.conv1.weight
    # for some reason have to get max values here....it gets completely screwd otherwise
    R_max = W[:,0,:,:].max() - W[:,0,:,:].min()
    G_max = W[:,1,:,:].max() - W[:,1,:,:].min()
    B_max = W[:,2,:,:].max() - W[:,2,:,:].min()
    tiles_imgc = tile_and_print(W,4,16, padding = 1)
    # tiles_imgc = tiles_imgc
    R, G, B = tiles_imgc[0,:,:], tiles_imgc[1,:,:], tiles_imgc[2,:,:]
    R = (R - R.min()) / R_max
    # R = R / R_max 
    G = (G - G.min()) / G_max
    # G = G / G_max
    B = (B - B.min()) / B_max
    # B = B / B_max

    tiles_imgc[0,:,:] = R
    tiles_imgc[1,:,:] = G
    tiles_imgc[2,:,:] = B 
    tiles_imgc = tiles_imgc.permute(1,2,0)
    tiles = tiles_imgc.cpu().detach().numpy()

    plt.figure(figsize = (20,200))
    plt.imshow(tiles, interpolation='bilinear')
    plt.show()
    plt.savefig(f'./plots/filt_{FILE}{int(time.time())}.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# calculate mean and stf of CIFAR....use sanketnet importer...use np for mean calc
def cifar_describe():
    paths = [
        "./resources/cifar/cifar-10-batches-py/data_batch_1",
        "./resources/cifar/cifar-10-batches-py/data_batch_2",
        "./resources/cifar/cifar-10-batches-py/data_batch_3",
        "./resources/cifar/cifar-10-batches-py/data_batch_4",
        "./resources/cifar/cifar-10-batches-py/data_batch_5"
    ]
    data = SANKETNET.DataImport.cifar(paths[0], mode="regression")[0]  # images coming in as vectors
    data = np.vstack((data, SANKETNET.DataImport.cifar(paths[1], mode="regression")[0]))
    data = np.vstack((data, SANKETNET.DataImport.cifar(paths[2], mode="regression")[0]))
    data = np.vstack((data, SANKETNET.DataImport.cifar(paths[3], mode="regression")[0]))
    data = np.vstack((data, SANKETNET.DataImport.cifar(paths[4], mode="regression")[0]))
    n = data.size
    data = data / 255
    data = data.transpose(1, 0)
    data = data.reshape(3, int(n / 3), 1)
    m = data.mean(axis=1).reshape(3)
    s = data.std(axis=1).reshape(3)
    assert m.size == 3
    print('mean', m, 'std', s)
    return list(m), list(s)


# taken from tutorial....SANKETNET hothelper doesn't work with tensors :(

def target_transform_cifar(y):
    # return torch.zeros(
    #     10, dtype=torch.float).scatter_(dim=-1, index=torch.tensor(y), value=1
    #                                     )
    return y
## Data import / load
# m, s = cifar_describe()
m, s = [0.49139956, 0.4821574,  0.44653055], [0.24703272, 0.24348505, 0.26158777]
# m, s = [0.5, 0.5,  0.5], [0.5, 0.5, 0.5]
def transform_cifar(img):
    img = transforms.Resize((224, 224))(img)  # must curry arg
    img = transforms.ToTensor()(img)  # 0 to 1 tensor
    img = transforms.Normalize(mean=m, std=s)(img)
    return img


cifar = datasets.CIFAR10(
    './resources/cifar/',
    True,
    transform_cifar,
    # target_transform_cifar,
    download=True
)

cifar_test = datasets.CIFAR10(
    './resources/cifar/',
    False,
    transform_cifar,
    # target_transform_cifar,
    download=True
)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        m = (x * torch.tanh(F.softplus(x)))  # https://arxiv.org/abs/1908.08681
        return m

# q if i wanted to just take weights from another model, for the fully connected part i'd have to
#  use the same batch size and number of features right? sounds kinds dumn...
class TorchNet(nn.Module):  
    def __init__(self, batchnorm = False):
        super(TorchNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (11, 11), padding=(5,))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(64, 128, (11, 11), padding=(5,))
        self.conv3 = nn.Conv2d(128, 128, (3, 3), padding=(1,))
        self.lin1 = nn.Linear(25088, 10)
        # self.lin2 = nn.Linear(392, 10)
        
        
    # q are we to average pool to 1 pixel?
    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 complete')
        assert x.shape == (hypes["BATCH"], 64, 224, 224), x.shape
        x = Mish()(x)
        x = self.max_pool(x)
        assert x.shape == (hypes["BATCH"], 64, 112, 112), x.shape
        # print('Maxpool complete')
        x = self.conv2(x)
        assert x.shape == (hypes["BATCH"], 128, 112, 112), x.shape
        # print('conv2 complete')
        x = Mish()(x)
        x = self.conv3(x)
        assert x.shape == (hypes["BATCH"], 128, 112, 112), x.shape
        # print('conv3 complete')
        x = Mish()(x)
        x = self.avg_pool(x)
        assert x.shape == (hypes["BATCH"], 128, 14, 14), x.shape
        # print('avgpool complete')
        x = x.view(hypes["BATCH"], 25088)
        y = self.lin1(x)
        # print('lin1 complete')
        # x = nn.ReLU()(x)
        # x = self.lin2(x)
        # print('lin2 complete')
        
        return y


def batches_loop(loader, model, criterion, optimizer, is_val=False):
    batch_count = 0
    loss_total = 0
    count_correct = 0
    # for x in loader:
    #     y = torch.ones(1, dtype=int)
    #     x = x['image']
    for x, y in loader:
        
        x = x.to(device)
        y = y.to(device)
        batch_count+=1
        # assert x.shape[0] == hypes["BATCH"], x.shape
        if is_val:
    
            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_total += loss.item()
        else:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        if batch_count % 250 == 0:
            print(batch_count,' batches complete')
        y_hat_arg = y_hat.argmax(dim=-1)
        count_correct += (y == y_hat_arg).sum()
    
    accuracy = (count_correct / (batch_count * hypes["BATCH"]))    
    return loss_total, accuracy

def problem3_1():
    print(hypes)
    cifar_Loader = DataLoader(cifar, drop_last=True, batch_size=hypes['BATCH'])
    cifar_test_Loader = DataLoader(cifar_test,drop_last=True, batch_size=hypes['BATCH'])
    network = TorchNet()
    network.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=hypes["RATE"], momentum=hypes["MOMENTUM"])
    count_epoch = 0
    accuracies = []
    accuracies_test = []
    loss_testing = []
    loss_training = []
    # for epoch in range(2):  # loop over the dataset multiple times
    for epoch in range(hypes["EPOCHS"]):
        count_epoch+=1
        print('EPOCH:', count_epoch)
        network.train()
        training_batches = batches_loop(cifar_Loader, network,criterion,optimizer)
        loss_training.append(training_batches[0])
        network.eval()
        testing_batches = batches_loop(cifar_test_Loader,network, criterion,optimizer, True)
        loss_testing.append(testing_batches[0])
        accuracy = training_batches[1]
        accuracies.append(accuracy)
        accuracy_test = testing_batches[1]
        accuracies_test.append(accuracy_test)
        print('training_loss', training_batches[0], 'accuracy', accuracy)
        print('testing_loss', testing_batches[0], 'accuracy_test', accuracy_test)
        metrics = (loss_training,loss_testing,accuracies,accuracies_test)
    save_bin(f'{FILE}', network)
    save_bin(f'{FILE}_metrics', metrics)
    filter_visual(network)
    return network,metrics


# batch norm is z-score norm per mini batch,possibly using whole pop stats, not sure, but def in testing use whole pop stats...
# though i thought it just turns off during testing.......no BIAS in conv layer, bc BN layer adds bias...
# 

class TorchNet_bn(nn.Module):  
    def __init__(self, batchnorm = False):
        super(TorchNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (11, 11), padding=(5,),bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(64, 128, (11, 11), padding=(5,),bias=False)
        self.conv3 = nn.Conv2d(128, 128, (3, 3), padding=(1,),bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.lin1 = nn.Linear(25088, 10)
        # self.lin2 = nn.Linear(392, 10)
        
        
    # q are we to average pool to 1 pixel?
    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 complete')
        assert x.shape == (hypes["BATCH"], 64, 224, 224), x.shape
        x = self.bn1(x)
        x = Mish()(x)
        x = self.max_pool(x)
        assert x.shape == (hypes["BATCH"], 64, 112, 112), x.shape
        # print('Maxpool complete')
        x = self.conv2(x)
        assert x.shape == (hypes["BATCH"], 128, 112, 112), x.shape
        # print('conv2 complete')
        x = self.bn2(x)
        x = Mish()(x)
        x = self.conv3(x)
        assert x.shape == (hypes["BATCH"], 128, 112, 112), x.shape
        # print('conv3 complete')
        x = self.bn3(x)
        x = Mish()(x)
        x = self.avg_pool(x)
        assert x.shape == (hypes["BATCH"], 128, 14, 14), x.shape
        # print('avgpool complete')
        x = x.view(hypes["BATCH"], 25088)
        y = self.lin1(x)
        # print('lin1 complete')
        # x = nn.ReLU()(x)
        # x = self.lin2(x)
        # print('lin2 complete')
        
        return y

def problem3_2():
    print(hypes)
    cifar_Loader = DataLoader(cifar, drop_last=True, batch_size=hypes['BATCH'])
    cifar_test_Loader = DataLoader(cifar_test,drop_last=True, batch_size=hypes['BATCH'])
    network = TorchNet_bn()
    network.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=hypes["RATE"], momentum=hypes["MOMENTUM"])
    count_epoch = 0
    accuracies = []
    accuracies_test = []
    loss_testing = []
    loss_training = []
    # for epoch in range(2):  # loop over the dataset multiple times
    for epoch in range(hypes["EPOCHS"]):
        count_epoch+=1
        print('EPOCH:', count_epoch)
        network.train()
        training_batches = batches_loop(cifar_Loader, network,criterion,optimizer)
        loss_training.append(training_batches[0])
        network.eval()
        testing_batches = batches_loop(cifar_test_Loader,network, criterion,optimizer, True)
        loss_testing.append(testing_batches[0])
        accuracy = training_batches[1]
        accuracies.append(accuracy)
        accuracy_test = testing_batches[1]
        accuracies_test.append(accuracy_test)
        print('training_loss', training_batches[0], 'accuracy', accuracy)
        print('testing_loss', testing_batches[0], 'accuracy_test', accuracy_test)
        metrics = (loss_training,loss_testing,accuracies,accuracies_test)
    save_bin(f'{FILE}', network)
    save_bin(f'{FILE}_metrics', metrics)
    filter_visual(network)
    return network,metrics


class SuperNet(nn.Module):  
    def __init__(self, batchnorm = False):
        super(SuperNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (7, 7), padding=(3,),bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), padding=(2,),bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, (5, 5), padding=(2,),bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        # RELU

        self.conv4 = nn.Conv2d(64,128,(3,3), padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.avg_pool2 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.conv5 = nn.Conv2d(128,256,(3,3), padding=1,bias=False)
        self.bn5= nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,256,(1,1), padding=0,bias=False)
        self.bn6 = nn.BatchNorm2d(256)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256,64,(1,1), padding=0,bias=False)
        self.bn7 = nn.BatchNorm2d(64)

        self.lin1 = nn.Linear(64*14*14, 64)
        self.lin2 = nn.Linear(64, 10)

        
        
    # q are we to average pool to 1 pixel?
    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 complete')
        assert x.shape == (hypes["BATCH"], 16, 224, 224), x.shape
        x = self.bn1(x)
        x = Mish()(x)

        x = self.conv2(x)
        assert x.shape == (hypes["BATCH"], 32, 224, 224), x.shape
        # print('conv2 complete')
        x = self.bn2(x)
        x = Mish()(x)

        x = self.avg_pool1(x)

        x = self.conv3(x)
        assert x.shape == (hypes["BATCH"], 64, 112, 112), x.shape
        # print('conv2 complete')
        x = self.bn3(x)
        x = Mish)(x)

        x = self.conv4(x)
        assert x.shape == (hypes["BATCH"], 128, 112, 112), x.shape
        # print('conv2 complete')
        x = self.bn4(x)
        x = Mish()(x)

        x = self.avg_pool2(x)

        x = self.conv5(x)
        assert x.shape == (hypes["BATCH"], 256, 28, 28), x.shape
        # print('conv2 complete')
        x = self.bn5(x)
        x = Mish()(x)

        x = self.conv6(x)
        assert x.shape == (hypes["BATCH"], 256, 28, 28), x.shape
        # print('conv2 complete')
        x = self.bn6(x)
        x = Mish()(x)

        x = self.max_pool(x)

        x = self.conv7(x)
        assert x.shape == (hypes["BATCH"], 64, 14, 14), x.shape
        # print('conv2 complete')
        x = self.bn7(x)
        x = Mish()(x)
        
        x = x.view(hypes["BATCH"], 64*14*14)
        x = self.lin1(x)
        x = Mish()(x)

        y = self.lin2(x)
        
        return y

def problem3_3():
    print(hypes)
    cifar_Loader = DataLoader(cifar, drop_last=True, batch_size=hypes['BATCH'])
    cifar_test_Loader = DataLoader(cifar_test,drop_last=True, batch_size=hypes['BATCH'])
    network = SuperNet()
    network.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=hypes["RATE"], momentum=hypes["MOMENTUM"])
    count_epoch = 0
    accuracies = []
    accuracies_test = []
    loss_testing = []
    loss_training = []
    # for epoch in range(2):  # loop over the dataset multiple times
    for epoch in range(hypes["EPOCHS"]):
        count_epoch+=1
        print('EPOCH:', count_epoch)
        network.train()
        training_batches = batches_loop(cifar_Loader, network,criterion,optimizer)
        loss_training.append(training_batches[0])
        network.eval()
        testing_batches = batches_loop(cifar_test_Loader,network, criterion,optimizer, True)
        loss_testing.append(testing_batches[0])
        accuracy = training_batches[1]
        accuracies.append(accuracy)
        accuracy_test = testing_batches[1]
        accuracies_test.append(accuracy_test)
        print('training_loss', training_batches[0], 'accuracy', accuracy)
        print('testing_loss', testing_batches[0], 'accuracy_test', accuracy_test)
        metrics = (loss_training,loss_testing,accuracies,accuracies_test)
    save_bin(f'{FILE}', network)
    save_bin(f'{FILE}_metrics', metrics)
    filter_visual(network)
    return network, metrics
    
## Eventually
# batch Norm layers
# gpu condition checking...
# optimizer / parameter search...
# cli customizing (will need if using colab)

##########
### CLI code
########
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--EPOCHS", type=int, default=100, help="number of epochs")
    parser.add_argument("--RATE",type=float,default= .01,help="learning rate")
    parser.add_argument("--BATCH", default=64, type=int, help="backend mode")
    parser.add_argument("--RC", type=float,default=0, help="regularizer")
    parser.add_argument("--MOMENTUM", default=0,type=float, help="Momentum")
    parser.add_argument("--FILE", type=str, help="Bin file name")
    args = parser.parse_known_args()[0]

    def command(func):
        global hypes
        hypes['EPOCHS'] = args.EPOCHS
        hypes['MOMENTUM'] = args.MOMENTUM
        hypes['BATCH'] = args.BATCH
        hypes['RATE'] = args.RATE
        hypes['RC'] = args.RC
        global FILE
        FILE = args.FILE
        return func()
    # print(hypes)
    ## name/main is for a functioncall that only should be called when module is main mofule running,
    # otherwise would run every time imported.

    FUNCTION_MAP = {'problem3_1' : problem3_1,
                    'problem2b' : problem2b,
                    'problem3_2' : problem3_2,
                    'problem3_3' : problem3_3

                    }
    parser.add_argument('--RUN', choices=FUNCTION_MAP.keys())
    args = parser.parse_args()
    command(FUNCTION_MAP[args.RUN])

