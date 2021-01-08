import torch
import math

class teca_deeplab_ar_detect_internals:

    # Implementation of Google's Deeplab-V3-Plus
    # source: https://arxiv.org/pdf/1802.02611.pdf
    class Bottleneck(torch.nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes,
                     stride=1, rate=1, downsample=None):

            super(teca_deeplab_ar_detect_internals.Bottleneck,
                  self).__init__()
            self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(planes)
            self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, dilation=rate,
                                   padding=rate, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(planes)
            self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1,
                                   bias=False)
            self.bn3 = torch.nn.BatchNorm2d(planes * 4)
            self.relu = torch.nn.ReLU(inplace=True)
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

    class ResNet(torch.nn.Module):
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
            self.conv1 = torch.nn.Conv2d(nInputChannels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
                downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(planes * block.expansion),
                    )

            layers = []
            layers.append(
                block(self.inplanes, planes, stride, rate, downsample)
                )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return torch.nn.Sequential(*layers)

        def _make_MG_unit(self, block, planes,
                          blocks=[1, 2, 4], stride=1, rate=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(planes * block.expansion),
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

            return torch.nn.Sequential(*layers)

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
                if isinstance(m, torch.nn.Conv2d):
                    torch.torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    class ASPP_module(torch.nn.Module):
        def __init__(self, inplanes, planes, rate):

            super(teca_deeplab_ar_detect_internals.ASPP_module,
                  self).__init__()
            if rate == 1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = rate
            self.atrous_convolution = torch.nn.Conv2d(
                inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=rate, bias=False
                )
            self.bn = torch.nn.BatchNorm2d(planes)
            self.relu = torch.nn.ReLU()

            self._init_weight()

        def forward(self, x):
            x = self.atrous_convolution(x)
            x = self.bn(x)

            return self.relu(x)

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    class DeepLabv3_plus(torch.nn.Module):
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

            self.relu = torch.nn.ReLU()

            self.global_avg_pool = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU()
            )

            self.conv1 = torch.nn.Conv2d(1280, 256, 1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(256)

            # adopt [1x1, 48] for channel reduction.
            self.conv2 = torch.nn.Conv2d(256, 48, 1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(48)

            self.last_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    304, 256, kernel_size=3, stride=1,
                    padding=1, bias=False
                ),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    256, 256, kernel_size=3, stride=1,
                    padding=1, bias=False
                ),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        def forward(self, input):
            x, low_level_features = self.resnet_features(input)
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.global_avg_pool(x)

            x5 = torch.nn.functional.interpolate(
                x5, size=x4.size()[2:], mode='bilinear',
                align_corners=True
                )

            x = torch.cat((x1, x2, x3, x4, x5), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = torch.nn.functional.interpolate(
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

            x = torch.nn.functional.interpolate(
                x, size=input.size()[2:], mode='bilinear',
                align_corners=True
                )

            return x

        def freeze_bn(self):
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        def __init_weight(self):
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


