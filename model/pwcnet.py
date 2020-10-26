# Copyright (C) 2018 sniklaus. 
# All rights reserved. 
# Licensed under the GNU General Public License v3.0 (https://github.com/sniklaus/pytorch-pwc/blob/master/LICENSE)

import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer
import math
import torchvision.models as models
import torchvision.transforms as transforms

from model.correlation import correlation # the custom cost volume layer

backward_grid = {}
backward_partial = {}


def Backward(x, flow):
    if str(flow.size()) not in backward_grid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, flow.size(3), device=x.device).view(1, 1, 1, flow.size(3)).expand(flow.size(0), -1, flow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, flow.size(2), device=x.device).view(1, 1, flow.size(2), 1).expand(flow.size(0), -1, -1, flow.size(3))

        backward_grid[str(flow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], dim=1).cuda()

    if str(flow.size()) not in backward_partial:
        backward_partial[str(flow.size())] = flow.new_ones([ flow.size(0), 1, flow.size(2), flow.size(3) ])

    flow = torch.cat([ flow[:, 0:1, :, :] / ((x.size(3) - 1.0) / 2.0), flow[:, 1:2, :, :] / ((x.size(2) - 1.0) / 2.0) ], dim=1)
    grid = (backward_grid[str(flow.size())] + flow).permute(0, 2, 3, 1)

    x = torch.cat([ x, backward_partial[str(flow.size())] ], 1)

    output = torch.nn.functional.grid_sample(input=x, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    mask = output[:, -1:, :, :]
    mask[mask > 0.999] = 1.0; mask[mask < 1.0] = 0.0

    return (output[:, :-1, :, :] * mask).contiguous()

##########################################################


class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()

        ##########################################################################################
        # Feature Pyramid Extractor
        ##########################################################################################
        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                input_fn = 3

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=input_fn, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    # torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    # torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    # torch.nn.BatchNorm2d(96),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(96),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(96),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    # torch.nn.BatchNorm2d(196),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(196),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(196),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

            def forward(self, x):
                # we assume the color image is already in the range [0, 1]
                assert torch.max(x[:, :3, :, :]) <= 1.0, "input image is not within the range [0, 1], but rather within [0, {}]".format(torch.max(x))
                tensorOne = self.moduleOne(x)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]

        ##########################################################################################
        # Decoder
        ##########################################################################################
        feature_sizes = [ None, None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81,   None]
        scales        = [ None, None, None,      5.0,       2.5,       1.25,       0.625     ]
        dd = np.cumsum([128, 128, 96, 64, 32])

        class Decoder(torch.nn.Module):
            def __init__(self, lvl):
                super(Decoder, self).__init__()

                previous = feature_sizes[lvl + 1]
                current  = feature_sizes[lvl]

                if lvl < 6: 
                    self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                    self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=previous + dd[4], out_channels=2, kernel_size=4, stride=2, padding=1)
                    self.dblBackward = scales[lvl + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current,         out_channels=128, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current + dd[0], out_channels=128, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current + dd[1], out_channels=96, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(96),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current + dd[2], out_channels=64, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current + dd[3], out_channels=32, kernel_size=3, stride=1, padding=1),
                    # torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    #torch.nn.Dropout2d(p=0.2, inplace=False)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=current + dd[4], out_channels=2, kernel_size=3, stride=1, padding=1) # predict flow
                )

            def forward(self, first, second, object_prev):
                flow = None
                features = None

                if object_prev is None:
                    flow = None
                    features = None

                    cost_volume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(first=first, second=second), negative_slope=0.1, inplace=False)
                    
                    features = cost_volume

                else:
                    flow = self.moduleUpflow(object_prev['flow'])
                    features = self.moduleUpfeat(object_prev['features'])

                    assert flow is not None
                    assert features is not None

                    second_warped = Backward(x=second, flow=flow * self.dblBackward)
                    cost_volume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(first=first, second=second_warped), negative_slope=0.1, inplace=False)

                    features = torch.cat([ cost_volume, first, flow, features ], 1)

                assert features is not None

                features = torch.cat([ self.moduleOne(features), features ], 1)
                features = torch.cat([ self.moduleTwo(features), features ], 1)
                features = torch.cat([ self.moduleThr(features), features ], 1)
                features = torch.cat([ self.moduleFou(features), features ], 1)
                features = torch.cat([ self.moduleFiv(features), features ], 1)
                
                flow = self.moduleSix(features)

                assert flow is not None

                return {
                    'flow': flow,
                    'features': features
                }

        ##########################################################################################
        # Refiner
        ##########################################################################################
        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=feature_sizes[2] + dd[4], out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    # torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    # torch.nn.BatchNorm2d(96),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    # torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    # torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #torch.nn.Dropout2d(p=0.2, inplace=False),
                    
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1) # predict flow
                )

            def forward(self, x):
                return self.moduleMain(x)

        # Define PWCNet
        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

    def forward(self, first, second):
        first = first[:, :3, :, :]
        second = second[:, :3, :, :]
            
        pyramid_first  = self.moduleExtractor(first)
        pyramid_second = self.moduleExtractor(second)

        object_estimate = self.moduleSix(pyramid_first[-1], pyramid_second[-1], None)
        flow6 = object_estimate['flow']

        object_estimate = self.moduleFiv(pyramid_first[-2], pyramid_second[-2], object_estimate)
        flow5 = object_estimate['flow']
        
        object_estimate = self.moduleFou(pyramid_first[-3], pyramid_second[-3], object_estimate)
        flow4 = object_estimate['flow']
        
        object_estimate = self.moduleThr(pyramid_first[-4], pyramid_second[-4], object_estimate)
        flow3 = object_estimate['flow']

        object_estimate = self.moduleTwo(pyramid_first[-5], pyramid_second[-5], object_estimate)
        flow2 = object_estimate['flow'] + self.moduleRefiner(object_estimate['features'])

        features2 = object_estimate['features']

        return flow2, flow3, flow4, flow5, flow6, features2
