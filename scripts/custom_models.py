
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Part 2 - Model Construction and Initialization.ipynb

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import imageio
from skimage import io, transform
import torchvision
from torchvision import transforms

from scripts.dataloader import Dataset, Transforms, Resize, ToTorch, Sampler, collate, DataLoader
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def flatten(x):
    return x.view(x.shape[0], -1)

def simulate_fc_output(x, n):
    return torch.rand((x.shape[0], n))



class MixedInputModel(nn.Module):
    def __init__(self, cnn_model,  tabular_model, mixed_model):
        super(MixedInputModel, self).__init__()

        self.cnn_model = cnn_model
        self.tabular_model = tabular_model
        self.mixed_model = mixed_model


    def forward(self, x):
        #unpack the x_batch tuple into the image and tabular components
        x_image, x_tab = x
        x_image = x_image.float()
        x_tab = x_tab.float()

        #run each component seperately through their respective models
        cnn_output = self.cnn_model(x_image)
        tabular_output = self.tabular_model(x_tab)

        #concatenate the outputs from both networks and pass it through the mixed model output
        concat_outputs = torch.cat((cnn_output, tabular_output), dim = 1)
        mixed_model_output = self.mixed_model(concat_outputs)

        return(mixed_model_output)

class TabularModel(nn.Module):
    def __init__(self, layer_sizes):
        super(TabularModel, self).__init__()

        layers = []

        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())


        self.model = nn.Sequential(*layers[:-1]) #ignore the last nn.ReLU

    def forward(self, x):
        return self.model(x)


# class Lambda(nn.Module):
#     def __init__(self, func):
#         super(Lambda, self).__init__()
#         self.func = func

#     def forward(self, x):
#         return self.func(x)


# def flatten(x):
#     return x.view(x.shape[0], -1)


class CNNModel(nn.Module):
    h = [5, 7, 10, 14] #hidden layer channels
    def __init__(self, img_channels, img_size):
        super(CNNModel, self).__init__()
        self.img_channels, self.size = img_channels, img_size

        current_channels = img_channels
        output_size = img_size

        cnn_model_components = []
        for new_channels in self.h:
            layer, output_size = self.get_cnn_layer(current_channels, new_channels, 3, 1, 2, output_size)
            current_channels = new_channels
            cnn_model_components.append(layer)

        while output_size >5:
            layer, output_size = self.get_cnn_layer(self.h[-1], self.h[-1], 3, 1, 2, input_size = output_size)

        cnn_model_components.append(nn.AdaptiveAvgPool2d(1))
        cnn_model_components.append(Lambda(flatten))

        fcc_model_components = nn.Sequential(nn.Linear(self.h[-1], 20), nn.ReLU(),
                                            nn.Linear(20, 10), nn.ReLU(),
                                            nn.Linear(10, 1))

        self.model = nn.Sequential(*cnn_model_components, fcc_model_components)




    def forward(self, x):
        return self.model(x)

    def get_cnn_layer(self, inp_chs, out_chs, kernel_size = 3, padding = 1, stride = 2, input_size = None):
        """
        This function acts as a default for


        We can keep track of the final size of our network based on the initial size.  The formula for output
        size is the floor of O = ((Input_size + 2*padding - dilation*(kernel_size-1) -1)/stride) + 1.  For instance, with
        input size = 256, kernel_size = 3, padding = 2 and stride = 2, we get (256+2*2-1*(3-1)-1)/2 = 127.5, whose floor
        is 127.  We therefore expect the output height and width to be 127

        """

        cnn_layer = nn.Conv2d(inp_chs, out_chs, kernel_size, stride, padding,1)

        if input_size is None: return cnn_layer
        else:
            new_size = ((input_size + 2*padding - 1*(kernel_size - 1) -1)/stride + 1)//1
            return cnn_layer, new_size







class CustomResnet(nn.Module):
    def __init__(self, base_model, connected_layer_sizes):
        super(CustomResnet, self).__init__()
        #self.base_mode = base_model
        #self.model_outputs = model_outputs
        self.connected_layer_sizes = connected_layer_sizes

        connected_layers = []
        for i in range(len(connected_layer_sizes)-1):
            h1 = connected_layer_sizes[i]
            h2 = connected_layer_sizes[i+1]

            connected_layers.append(nn.Linear(h1, h2))
            connected_layers.append(nn.ReLU())

        connected_model = nn.Sequential(*connected_layers)


        self.model = nn.Sequential(base_model, connected_model)

    def forward(self, x):
        return self.model(x)
