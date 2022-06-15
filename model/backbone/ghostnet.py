"""
2020.06.09-Changed for building GhostNet
Huawei Technologies Co., Ltd. <foss@huawei.com>
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang,
Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
and https://github.com/rwightman/pytorch-image-models
"""
import logging
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.activation import act_layers

'''
    华为提出的轻量骨干网路,用低成本的线性变换得到ghost feature
    同样提供了预训练权重
'''

def get_url(width_mult=1.0):
    if width_mult == 1.0:
        return "https://raw.githubusercontent.com/huawei-noah/CV-Backbones/master/ghostnet_pytorch/models/state_dict_73.98.pth"  # noqa E501
    else:
        logging.info("GhostNet only has 1.0 pretrain model. ")
        return None


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        activation="ReLU",
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        # channel-wise的全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1x1卷积,得到一个维度更小的一维向量
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # 送入激活层
        self.act1 = act_layers(activation)
        # 再加上一个1x1 conv,使得输出长度还原回通道数
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # 用刚得到的权重乘以原输入
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, activation="ReLU"):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layers(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation="ReLU"
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        # 确定特征层减少的比例,init_channels是标准卷积操作得到
        init_channels = math.ceil(oup / ratio)
        # new_channels是利用廉价操作得到的
        new_channels = init_channels * (ratio - 1)

        # 标准的conv BN activation层,注意conv是point-wise conv的1x1卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )

        # ghostNet的核心,用廉价的线性操作来生成相似特征图
        # 关键在于groups数为init_channels,则说明每个init_channel都对应一层conv
        # 输出的通道数是输入的ratio-1倍,输入的每一个channel会有ratio-1组参数
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            # BN和AC操作
            nn.BatchNorm2d(new_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # new_channel和init_channel是并列的关系,拼接在一起形成新的输出
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        activation="ReLU",
        se_ratio=0.0,
    ):
        super(GhostBottleneck, self).__init__()
        # 可以选择是否加入SE module
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, activation=activation)

        # Depth-wise convolution
        # 对于stride=2的版本(或者你自己选择添加更大的Stride),两个GhostModule中间增加DW卷积
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        # 最后的输出不添加激活函数层
        self.ghost2 = GhostModule(mid_chs, out_chs, activation=None)

        # shortcut
        # 最后的跳连接,如果in_chs等于out_chs则直接执行element-wise add
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        # 如果不相等,则使用深度可分离卷积使得feature map的大小对齐
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        # 保留identity feature,稍后进行连接
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # 如果stride>1则加入Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(
        self,
        width_mult=1.0,
        out_stages=(4, 6, 9),
        activation="ReLU",
        pretrain=True,
        act=None,
    ):
        super(GhostNet, self).__init__()
        assert set(out_stages).issubset(i for i in range(10))
        self.width_mult = width_mult
        self.out_stages = out_stages
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t,   c,  SE, s
            # stage1
            [[3, 16, 16, 0, 1]],  # 0
            # stage2
            [[3, 48, 24, 0, 2]],  # 1
            [[3, 72, 24, 0, 1]],  # 2  1/4
            # stage3
            [[5, 72, 40, 0.25, 2]],  # 3
            [[5, 120, 40, 0.25, 1]],  # 4  1/8
            # stage4
            [[3, 240, 80, 0, 2]],  # 5
            [
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
            ],  # 6  1/16
            # stage5
            [[5, 672, 160, 0.25, 2]],  # 7
            [
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
            ],  # 8
        ]
        #  ------conv+bn+act----------# 9  1/32

        self.activation = activation
        if act is not None:
            warnings.warn(
                "Warning! act argument has been deprecated, " "use activation instead!"
            )
            self.activation = act

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = act_layers(self.activation)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        activation=self.activation,
                        se_ratio=se_ratio,
                    )
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(
            nn.Sequential(
                ConvBnAct(input_channel, output_channel, 1, activation=self.activation)
            )
        )  # 9

        self.blocks = nn.Sequential(*stages)

        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        for i in range(10):
            x = self.blocks[i](x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "conv_stem" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = get_url(self.width_mult)
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict, strict=False)
