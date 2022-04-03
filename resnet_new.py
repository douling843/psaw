import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import cmath
import matplotlib.pyplot as plt
# from mmdet.apis import init_detector, inference_detector



BN_MOMENTUM = 0.1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):
    def __init__(self, block, layers, head_conv, num_classes):
        super(PoseResNet, self).__init__()
        self.inplanes = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        # self.final_layer = []
        self.se1024 = se_module(1024)
        self.se512 = se_module(512)
        self.cbam1 = CBAM(1024)
        self.cbam2 = CBAM(512)
        self.pwam2048 = PWAM(2048)
        self.pwam1024 = PWAM(1024)
        self.pwam512 = PWAM(512)
        self.pwam256 = PWAM(256)
        self.se = se_module(512)
        self.simam = simam_module()
        # for resnet-50，resnet-101
        #         self.dec_c2 = CombinationModule(512, 256, group_norm=True)
        #         self.dec_c3 = CombinationModule(1024, 512,group_norm=True)
        #         self.dec_c4 = CombinationModule(2048, 1024, group_norm=True)

        self.spp = SPP(64 * block.expansion, 64 * block.expansion)

        #         if head_conv > 0:
        #             # heatmap layers
        #             self.hmap = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
        #                                       nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
        #                                       nn.ReLU(inplace=True),
        #                                       nn.Conv2d(head_conv, num_classes, kernel_size=1))
        #             self.hmap[-1].bias.data.fill_(-2.19)
        #             # regression layers
        #             self.regs = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
        #                                       nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
        #                                       nn.ReLU(inplace=True),
        #                                       nn.Conv2d(head_conv, 2, kernel_size=1))
        #             self.w_h_ = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
        #                                       nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
        #                                       nn.ReLU(inplace=True),
        #                                       nn.Conv2d(head_conv, 2, kernel_size=1))

        # block.expension代表通道扩大倍数，对于resnet18，expension为1，对于resnet50和101，expension为4
        # for resnet-18
        self.dec_c2 = CombinationModule(128 * block.expansion, 64 * block.expansion, group_norm=True)
        self.dec_c3 = CombinationModule(256 * block.expansion, 128 * block.expansion, group_norm=True)
        self.dec_c4 = CombinationModule(512 * block.expansion, 256 * block.expansion, group_norm=True)

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(64 * block.expansion, head_conv, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(head_conv),
                                      # BN not used in the paper, but would help stable training
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(64 * block.expansion, head_conv, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(head_conv),
                                      # BN not used in the paper, but would help stable training
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1))
            self.w_h_ = nn.Sequential(nn.Conv2d(64 * block.expansion, head_conv, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(head_conv),
                                      # BN not used in the paper, but would help stable training
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1))
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            # regression layers
            self.regs = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
            self.w_h_ = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=padding,
                                             output_padding=output_padding,
                                             bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        feat = []
        feat.append(x)  # C0
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # feat.append(x1)       # C1
        x1 = self.maxpool(x1)

        x2 = self.layer1(x1)
        # feat.append(x2)       # C2
        x3 = self.layer2(x2)
        # feat.append(x3)       # C3
        x4 = self.layer3(x3)
        # feat.append(x4)       # C4
        x5 = self.layer4(x4)
        # feat.append(x5)       # C5
        
        x5 = self.pwam2048(x5)
        c4_combine = self.dec_c4(x5, x4)
        c4_combine = self.pwam1024(c4_combine)
        #c4_combine = self.cbam1(c4_combine)
        c3_combine = self.dec_c3(c4_combine, x3)
#         c3_combine = self.se(c3_combine)
#         c3_combine = self.cbam2(c3_combine)
        c3_combine = self.pwam512(c3_combine)             ##############
#         c3_combine = self.simam(c3_combine)
        c2_combine = self.dec_c2(c3_combine, x2)
        c2_combine = self.pwam256(c2_combine)
#         lena_1 = c2_combine[0, 55, :, :]*255
#         lena_1 = lena_1.data.cpu().numpy()
#         plt.axis('off')

#         plt.imshow(lena_1)
#         # ax = plt.gca()
#         # plt.rcParams['savefig.dpi'] = 128
#         # plt.rcParams['figure.dpi'] = 128
#         # # plt.colorbar()
#         # plt.savefig('ax', box_inches='tight', pad_inches=0)
#         plt.show()
        
        
        x = c2_combine
        x = self.spp(x)
        # x = self.deconv_layers(x)
        out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
        #         if self.training:
        return out

    #         else:
    #             return out,c4_combine,c3_combine,c2_combine
    def init_weights(self, num_layers, is_pretrain):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #             nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                torch.nn.init.xavier_normal_(m.weight.data)
            #                 if m.bias is not None:
            #                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if is_pretrain:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)


#         for m in self.deconv_layers.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.normal_(m.weight, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.hmap.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.constant_(m.bias, -2.19)
#         for m in self.regs.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)
#         for m in self.w_h_.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)
#                 nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def resnet_18():
    model = PoseResNet(BasicBlock, [2, 2, 2, 2], head_conv=64, num_classes=80)
    model.init_weights(18)
    return model


def get_pose_net(num_layers, head_conv, pretrained=True, num_classes=80):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes)
    model.init_weights(num_layers, is_pretrain=True)
    return model


class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(c_up),
                                          nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_up, x_low), 1))


#################################################################################################################    
# spp  
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9,13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


#################################################################################################################
# CBAM
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


################################   PWAM：Piecewise weighted attention model    ############################
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class PWAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(PWAM, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
                 Flatten(),
                 nn.Linear(gate_channels, gate_channels // reduction_ratio),
                 nn.ReLU(),
                 nn.Linear(gate_channels // reduction_ratio, gate_channels))

        
    def forward(self, x):
        ###############################F通道用方差##########################
        b, c, h, w = x.size()
        n = w * h - 1
        min_pool = -F.max_pool2d(-x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        x_minus_min_square = (x - min_pool).pow(2)
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        quan = x_minus_min_square / (2*(x_minus_mu_square.sum(dim=[2,3], keepdim=True)) / n + 0.0001)
        x_out = quan*x
        x_out = x*F.sigmoid(x_out)



           #########################################F通道用最大最小值
#         max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
# #         max_pool = self.mlp(max_pool).unsqueeze(2).unsqueeze(3)
#
#         min_pool = -F.max_pool2d(-x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
# #         min_pool = self.mlp(min_pool).unsqueeze(2).unsqueeze(3)
#
#         interval = (max_pool-min_pool)/3
#
#         max_value = x > (max_pool-interval)
#         max_value = 1.0*max_value
#         max_value_x = (1+interval)*max_value*x
#
#         min_value = x < (min_pool+interval)
#         min_value = 1.0*min_value
#         min_value_x = (1-interval)*min_value*x
#
#         middle_value = 1 - max_value - min_value
#         middle_value_x = middle_value*x
#
#         x_recovery = max_value_x + min_value_x + middle_value_x
#
#
#         x_out = x*F.sigmoid(x_recovery)
        
        ##############################F通道用范数时间消耗太大###################################
#         min_pool = -F.max_pool2d(-x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         x = x.data.cpu().numpy()
#         x_norm = np.linalg.norm(x, ord=2, axis=(2, 3), keepdims=True)
#         x_norm = torch.from_numpy(x_norm)
#         x_norm = x_norm.cuda()
#         x = torch.from_numpy(x)
#         x = x.cuda()

#         w = (x - min_pool)/(x_norm+0.0001)
#         c = (x.size(2)*x.size(3))**0.5
#         w = w*c
#         x_out = w*x
#         x_out = x * F.sigmoid(x_out)
        ###############################F通道用方差##########################
#         b, c, h, w = x.size()
#         n = w * h - 1
#         min_pool = -F.max_pool2d(-x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         x_minus_min_square = (x - min_pool).pow(2)
#         x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
#         quan = x_minus_min_square / (2*(x_minus_mu_square.sum(dim=[2,3], keepdim=True)) / n + 0.0001) 
#         x_out = quan*x
#         x_out = x*F.sigmoid(x_out) 
        

    
    
        
        ###################################################################
        max_value_a = torch.max(x_out, 1)[0].unsqueeze(1)  #         min_value_a = torch.min(x_out, 1)[0].unsqueeze(1)
#         mean_value_a = torch.mean(x_out, 1).unsqueeze(1)
        min_value_a = -torch.max(-x_out, 1)[0].unsqueeze(1)
#         ##############################################
        interval_a = (max_value_a-min_value_a )/3
        
        max_value = x_out > (max_value_a-interval_a)
        max_value = 1.0*max_value
        max_value_x = (1+interval_a)*max_value*x_out

        min_value = x_out < (min_value_a+interval_a)
        min_value = 1.0*min_value
        min_value_x = (1-interval_a)*min_value*x_out

        middle_value = 1 - max_value - min_value
        middle_value_x = middle_value*x_out

        x_recovery = max_value_x + min_value_x + middle_value_x
        x_out = x_out*F.sigmoid(x_recovery)
#         ############################################
#         b, c, h, w = x_out.size()
#         x_out = x_out.data.cpu().numpy()  
#         x_norm = np.linalg.norm(x_out, ord=2, axis=1, keepdims=True)

#         x_norm = torch.from_numpy(x_norm) 
#         x_norm = x_norm.cuda()
        
#         x_out = torch.from_numpy(x_out) 
#         x_out = x_out.cuda()
# #         min_value_a = min_value_a.expand(b, c, h, w)

    
    
    
#         t_mean = x_out.mean(dim=[1], keepdim=True)
#         x_minus_mu_square = (x_out - t_mean).pow(2)
#         fangcha = x_minus_mu_square.sum(dim=[1], keepdim=True)/(c-1)
# #         w2 = (x_out - t_mean).pow(2)
#         w = x_minus_mu_square /fangcha + 0.5
# #         x_out_x = w * x_out
#         x_out = x_out*F.sigmoid(w)
#         print("x_out.shape", x_out.size())
#         print("min_value_a.shape", min_value_a.size())
#         print("x_norm.shape", x_norm.size())
        
#         w = (x_out - min_value_a)/(x_norm/cmath.sqrt(c))
#         w2 = (x_out - mean_value_a).pow(2)

#         w = w2/x_norm
#         w = w*(c**0.5) + 0.5
#         x_out_x = w * x_out
#         x_out = x_out*F.sigmoid(x_out_x)
#         e_lambda = 1e-4
#         w = (x_out - min_value_a)/(mean_value_a + e_lambda)
#         x_out_x = (1 + w)*x_out
#          x_out = x_out*F.sigmoid(x_out_x)  
    
    
    
    
    ############################################################
#         mean_value_a = torch.mean(x_out, 1).unsqueeze(1) 
        
#         interval_a = (max_value_a-mean_value_a )/3
        
#         max_value = x_out > (max_value_a-interval_a)
#         max_value = 1.0*max_value
#         max_value_x = (1+interval_a)*max_value*x_out

#         min_value = x_out < (mean_value_a-interval_a)
#         min_value = 1.0*min_value
#         min_value_x = (1-interval_a)*min_value*x_out

#         middle_value = 1 - max_value - min_value
#         middle_value_x = middle_value*x_out

#         x_recovery = max_value_x + min_value_x + middle_value_x
#         x_out = x_out*F.sigmoid(x_recovery)
        

        ######################################################################
        
        
        
        
        return x_out 
    
 ################################### SE ################################   
class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def get_module_name():
        return "se"

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

############################   Simam    ###################################

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

########################################################################

if __name__ == '__main__':
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    net = get_pose_net(num_layers=50, head_conv=64)
    print(net)
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         m.register_forward_hook(hook)

    # y = net(torch.randn(2, 3, 384, 384))
    # print(y)
    # print(y.size())

