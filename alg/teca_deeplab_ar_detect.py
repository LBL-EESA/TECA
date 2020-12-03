import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class teca_deeplab_ar_detect_internals:

    # Implementation of Google's Deeplab-V3-Plus
    # source: https://arxiv.org/pdf/1802.02611.pdf
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes,
                     stride=1, rate=1, downsample=None):
            super(teca_deeplab_ar_detect_internals.Bottleneck,
                  self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, dilation=rate,
                                   padding=rate, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
            self.rate = rate

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

    class ResNet(nn.Module):
        def __init__(self, nInputChannels, block, layers, os=16):
            self.inplanes = 64
            super(teca_deeplab_ar_detect_internals.ResNet, self).__init__()
            if os == 16:
                strides = [1, 2, 2, 1]
                rates = [1, 1, 1, 2]
                blocks = [1, 2, 4]
            elif os == 8:
                strides = [1, 2, 1, 1]
                rates = [1, 1, 2, 2]
                blocks = [1, 2, 1]
            else:
                raise NotImplementedError

            # Modules
            self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0],
                                           stride=strides[0], rate=rates[0])
            self.layer2 = self._make_layer(block, 128, layers[1],
                                           stride=strides[1], rate=rates[1])
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=strides[2], rate=rates[2])
            self.layer4 = self._make_MG_unit(block, 512, blocks=blocks,
                                             stride=strides[3], rate=rates[3])

            self._init_weight()

        def _make_layer(self, block, planes, blocks, stride=1, rate=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )

            layers = []
            layers.append(
                block(self.inplanes, planes, stride, rate, downsample)
                )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def _make_MG_unit(self, block, planes,
                          blocks=[1, 2, 4], stride=1, rate=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )

            layers = []
            layers.append(
                block(self.inplanes, planes, stride,
                      rate=blocks[0]*rate, downsample=downsample)
                )
            self.inplanes = planes * block.expansion
            for i in range(1, len(blocks)):
                layers.append(
                    block(self.inplanes, planes,
                          stride=1, rate=blocks[i]*rate)
                    )

            return nn.Sequential(*layers)

        def forward(self, input):
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            low_level_feat = x
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x, low_level_feat

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    class ASPP_module(nn.Module):
        def __init__(self, inplanes, planes, rate):
            super(teca_deeplab_ar_detect_internals.ASPP_module,
                  self).__init__()
            if rate == 1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = rate
            self.atrous_convolution = nn.Conv2d(
                inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=rate, bias=False
                )
            self.bn = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU()

            self._init_weight()

        def forward(self, x):
            x = self.atrous_convolution(x)
            x = self.bn(x)

            return self.relu(x)

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    class DeepLabv3_plus(nn.Module):
        def __init__(self, nInputChannels=3, n_classes=21, os=16,
                     _print=True):
            if _print:
                sys.stdout.write("Constructing DeepLabv3+ model...\n")
                sys.stdout.write("Number of classes: {}\n".format(n_classes))
                sys.stdout.write("Output stride: {}\n".format(os))
                sys.stdout.write(
                    "Number of Input Channels: {}\n".format(nInputChannels)
                )
            super(teca_deeplab_ar_detect_internals.DeepLabv3_plus,
                  self).__init__()

            self.resnet_features = teca_deeplab_ar_detect_internals.ResNet(
                nInputChannels,
                teca_deeplab_ar_detect_internals.Bottleneck,
                [3, 4, 23, 3], os
                )

            # ASPP
            if os == 16:
                rates = [1, 6, 12, 18]
            elif os == 8:
                rates = [1, 12, 24, 36]
            else:
                raise NotImplementedError

            self.aspp1 = teca_deeplab_ar_detect_internals.ASPP_module(
                2048, 256, rate=rates[0])
            self.aspp2 = teca_deeplab_ar_detect_internals.ASPP_module(
                2048, 256, rate=rates[1])
            self.aspp3 = teca_deeplab_ar_detect_internals.ASPP_module(
                2048, 256, rate=rates[2])
            self.aspp4 = teca_deeplab_ar_detect_internals.ASPP_module(
                2048, 256, rate=rates[3])

            self.relu = nn.ReLU()

            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )

            self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(256)

            # adopt [1x1, 48] for channel reduction.
            self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(48)

            self.last_conv = nn.Sequential(
                nn.Conv2d(
                    304, 256, kernel_size=3, stride=1,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(
                    256, 256, kernel_size=3, stride=1,
                    padding=1, bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        def forward(self, input):
            x, low_level_features = self.resnet_features(input)
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.global_avg_pool(x)

            x5 = F.interpolate(
                x5, size=x4.size()[2:], mode='bilinear',
                align_corners=True
                )

            x = torch.cat((x1, x2, x3, x4, x5), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = F.interpolate(
                x,
                size=(
                    int(math.ceil(input.size()[-2]/4)),
                    int(math.ceil(input.size()[-1]/4))
                ),
                mode='bilinear',
                align_corners=True
                )

            low_level_features = self.conv2(low_level_features)
            low_level_features = self.bn2(low_level_features)
            low_level_features = self.relu(low_level_features)

            x = torch.cat((x, low_level_features), dim=1)
            x = self.last_conv(x)

            x = F.interpolate(
                x, size=input.size()[2:], mode='bilinear',
                align_corners=True
                )

            return x

        def freeze_bn(self):
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        def __init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class teca_deeplab_ar_detect(teca_pytorch_algorithm):
    """
    This algorithm detects Atmospheric Rivers using deep learning techniques
    derived from the DeepLabv3+ architecture. Given an input field of
    integrated vapor transport (IVT) magnitude, it calculates the probability
    of an AR event and stores it in a new scalar field named 'ar_probability'.
    """
    def __init__(self):
        super().__init__()

        self.set_input_variable("IVT")
        self.set_output_variable("ar_probability")

    def set_ivt_variable(self, var):
        """
        set the name of the variable containing the integrated vapor
        transport(IVT) magnitude field.
        """
        self.set_input_variable(var)

    def load_model(self, filename):
        """
        Load model from file system. In MPI parallel runs rank 0
        loads the model file and broadcasts it to the other ranks.
        """
        state_dict_deeplab = self.load_state_dict(filename)

        model = teca_deeplab_ar_detect_internals.DeepLabv3_plus(
            n_classes=1, _print=False)

        model.load_state_dict(state_dict_deeplab)

        self.set_model(model)

    def get_padding_sizes(self, div, dim):
        """
        given a divisor(div) and an input mesh dimension(dim)
        returns a tuple of values holding the number of values to
        add onto the low and high sides of the mesh to make the mesh
        dimension evely divisible by the divisor
        """
        # ghost cells in the y direction
        target_shape = div * np.ceil(dim / div)
        target_shape_diff = target_shape - dim

        pad_low = int(np.ceil(target_shape_diff / 2.0))
        pad_high = int(np.floor(target_shape_diff / 2.0))

        return pad_low, pad_high

    def preprocess(self, in_array):
        """
        resize the array to be a multiple of 64 in the y direction and 128 in
        the x direction amd convert to 3 channel (i.e. RGB image like)
        """
        nx_in = in_array.shape[1]
        ny_in = in_array.shape[0]

        # get the padding sizes to make the mesh evenly divisible by 64 in the
        # x direction and 128 in the y direction
        ng_x0, ng_x1 = self.get_padding_sizes(64.0, nx_in)
        ng_y0, ng_y1 = self.get_padding_sizes(128.0, ny_in)

        nx_out = ng_x0 + ng_x1 + nx_in
        ny_out = ng_y0 + ng_y1 + ny_in

        # allocate a new larger array
        out_array = np.zeros((1, 3, ny_out, nx_out), dtype=np.float32)

        # copy the input array into the center
        out_array[:, :, ng_y0 : ng_y0 + ny_in,
                  ng_x0 : ng_x0 + nx_in] = in_array

        # cache the padding info in order to extract the result
        self.ng_x0 = ng_x0
        self.ng_y0 = ng_y0
        self.nx_in = nx_in
        self.ny_in = ny_in

        return out_array

    def postprocess(self, out_tensor):
        """
        convert the tensor to a numpy array and extract the output data from
        the padded tensor. padding was added during preprocess.
        """
        # normalize the raw model output
        tmp = torch.sigmoid(out_tensor)

        # convert from torch tensor to numpy ndarray
        out_array = tmp.numpy()

        # extract the valid protion of the result
        out_array = out_array[:, :, self.ng_y0 : self.ng_y0 + self.ny_in,
                              self.ng_x0 : self.ng_x0 + self.nx_in]

        return out_array

    def report(self, port, md_in):
        """
        return the metadata decribing the data available for consumption.
        """
        md_out = super().report(port, md_in)

        arp_atts = teca_array_attributes(
            teca_float_array_code.get(),
            teca_array_attributes.point_centering,
            0, 'unitless', 'posterior AR flag',
            'the posterior probability of the presence '
            'of an atmospheric river')

        attributes = md_out["attributes"]
        attributes["ar_probability"] = arp_atts.to_metadata()
        md_out["attributes"] = attributes

        return md_out
