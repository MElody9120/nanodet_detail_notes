import torch
import torch.nn as nn

from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.scale import Scale


class SimpleConvHead(nn.Module):
    def __init__(
        self,
        num_classes, 
        input_channel,     # 输入的特征通道数
        feat_channels=256, # AGM内部的特征通道数
        stacked_convs=4,   # 使用四层卷积
        # 默认三个尺度,但是PAN中添加了额外层,配置文件可以看到是[8,16,32,64]
        strides=[8, 16, 32],  
        conv_cfg=None,
        # 使用group norm作为归一化层,效果优于BN
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        activation="LeakyReLU",
        # 配置文件中的默认参数是7
        reg_max=16,
        **kwargs
    ):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # range从0开始索引到stacked_convs-1
        for i in range(self.stacked_convs):
            # 第一层需要和输入对齐通道数,之后始终保持为feat_channels
            chn = self.in_channels if i == 0 else self.feat_channels
            # 分类分支
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
            # 回归分支
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        
        # 最后加上分类头
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        # 回归头的输出为4*(reg_max+1),解释见下文
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
        )
        # 用于缩放回归出的bbox的系数,因为AGM对不同尺寸的输入共享参数
        # AGM的输出直接用于标签分配,稍后在forward中可以看到还原检测框大小的操作
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    # 全部采用normal init,没什么好说的
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            # 对于来自PAN的每一层输入,计算class分支
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            # 计算regression分支
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            # 得到类别分数
            cls_score = self.gfl_cls(cls_feat)
            # 得到回归分布并进行缩放
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()
            # 拼接得到输出
            output = torch.cat([cls_score, bbox_pred], dim=1)
            # 追加到aux_pred后面
            outputs.append(output.flatten(start_dim=2))
        # 整理对齐维度,在之后的dsl_assigner中我们会详细介绍如何处理来自AGM和head的输出
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
