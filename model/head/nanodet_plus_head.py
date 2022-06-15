import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.
    Args:
        num_classes (int): Number of categories excluding the background
            category.不包括背景类,可以认为全部类的输出低于某个阈值视为背景
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
            刚送入检测头的通道数量,需要和PAN的输出通道数量保持一致
        feat_channels (int): Number of channels of the feature. 经过检测头卷积后的通道数
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs. 堆叠的卷积层数
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5. 
        strides (list[int]): Strides of input multi-level feature maps. 下采样步长
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
            AGM中使用GN,但是GN对于CPU型设备不友好,BN可以直接和卷积一同参数化而不需要额外计算
        reg_max (int): The maximal value of the discrete set. Default: 7. 
            参见GFL的论文,用于建模框的任意分布,而不是得到一个Dirac分布
            在第四部分的AGM中也介绍过,请查看往期博客
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """
    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        # 选择5x5的卷积大核
        kernel_size=5,
        # 输入特征相对原图像的下采样率,因为PAN中采用了extra_layer-
        # 从配置文件也可以看到这里实际有四层:[8,16,32,64]
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        # 必然是使用深度可分离卷积了,DepthwiseConvModule来自MMDetection,请直接看源码
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        # 第四部分介绍的动态分配器,稍后计算loss会使用到
        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        # 根据输出的框分布进行积分,得到最终的位置值
        self.distribution_project = Integral(self.reg_max)

        # 联合了分类和框的质量估计表示
        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        # 初始化参数中reg_max的由来,在对应模块中进行了详细的介绍
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        # IoU loss的一种改进,IoU loss家族还有CIoU/DIoU等
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()



    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            # 为每个stride的创建一个head,cls和reg共享这些参数
            cls_convs = self._buid_not_shared_head() 
            self.cls_convs.append(cls_convs)

        # 同样,为每个头增加gfl卷积
        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    # 每个位置需要num_classes个通道用于预测类别分数,还有4*(reg_max+1)来回归位置
                    # 用同一组卷积来获得,输出结果时再split成两份即可
                    self.num_classes + 4 * (self.reg_max + 1), 
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )



    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        # stacked_convs是参数中设定的卷积层数
        for i in range(self.stacked_convs):
            # 第一层要和PAN的输出对齐通道
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    # 加大小为卷积核一般的padding使得输入输出feat有相同尺寸
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs



    # 采用norm初始化
    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")



    # head的推理方法
    def forward(self, feats):
        # 有一个为了兼容onnx的方法
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        # 输出默认有4份,是一个list
        outputs = []
        # feats来自fpn,有多组,且组数和self.cls_cons/self.gfl_cls的数量需保持一致
        # 默认的参数设置是4组
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            # 对每组feat进行前向推理操作
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            # 所有head的输出会在展平后拼接成一个tensor,方便后处理
            # output是一个四维tensor,第一维长度为1(=batch size),批量训练或者推理时就是batch大小
            # 长为W宽为H(其实长宽相等)即feat的大小,高为80 + 4 * (reg_max+1)即cls和reg
            # 按照第三个维度展平,就是排成一个长度为W*H的tensor,另一个维度是输出的cls和reg
            outputs.append(output.flatten(start_dim=2)) # 变成1x112x(W*H)维了,80+4*8=112
        # 把不同head的输出交换一下维度排列顺序,全部拼在一起
        # 按照第三维拼接,就是1x112x2125(对于nanodet-m)
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs



    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output. head的输出
            gt_meta (dict): Ground truth information. 包含gt位置和标签的字典,还有原图像的数据
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.
            如果AGM还没有detach,会用AGM的输出进行标签分配

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        # 把gt相关的数据分离出来,这两个数据都是list,长度为batchsize的大小
        # 每个list中都包含了它们各自对应的图像上的gt和label
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        # 一会要传送到GPU上
        device = preds.device
        # 得到本次loss计算的batch数,pred是3维tensor,可以参见第一部分关于推理的介绍
        batch_size = preds.shape[0]

        # 对"img"信息取shape就可以得到图像的长宽,这里请看DataSet和DataLoader了解训练数据的详细格式
        # 所有图片都会在前处理中被resize成网络的输入大小,不足的则直接加zero padding
        input_height, input_width = gt_meta["img"].shape[2:]

        # 因为稍后要布置priors,这里要计算出feature map的大小
        # 如果修改了输入或者采样率,输入无法被stride整除,要取整
        # 默认是对齐的,应该为[40,40],[20,20],[10,10],[5,5]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]

        # get grid cells of one image
        # 在不同大小的stride上放置一组priors,默认四个检测头也就是四个不同尺寸的stride
        # 最后返回的是tensor维度是[batchsize,strideW*strideH,4]
        # 其中每一个都是[x,y,strideH,strideW]的结构,当featmap不是正方形的时候两个stride不相等
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]

        # 按照第二个维度拼接后的prior的维度是[batchsize,40x40+20x20+10x10+5x5=2125,4]
        # 其中四个值为[cx,cy,strideW,stridH]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        # 把预测值拆分成分类和框回归
        # cls_preds的维度是[batchsize,2125*class_num],reg_pred是[batchsize,2125*4*(reg_max+1)]
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )

        # 对reg_preds进行积分得到位置预测,reg_reds表示的是一条边的离散分布,
        # 积分就得到位置(对于离散来说就是加权求和),distribution_project()是Integral函数,稍后讲解
        # 乘以stride(在center_priors的最后两个位置)后,就得到[dl,dr,dt,db]在原图的长度了
        # dis_preds是[batchsize,2125,4],其中4为上述的中心到检测框四条边的距离
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        # 把[dl,dr,dt,db]根据prior的位置转化成框的左上角点和右下角点方便计算iou
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        # 如果启用了辅助训练模块,将用其结果进行标签分配
        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)

            # 可以去看multi_apply的实现,是一个稍微有点复杂的map()方法
            # 每次给一张图片进行分配,应该是为了避免显存溢出,代码写起来可读性也更高
            # 应该有并行优化,因此不用太担心效率问题
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        else:
            # use self prediction to assign
            # multi_apply将参数中的函数作用在后面的每一个可迭代对象上,一次处理批量数据
            # 因为target_assign_single_img一次只能分配一张图片
            # 并且由于显存限制,有时候无法一次处理整个batch
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )

        # 根据分配结果计算loss,这个函数稍后会介绍
        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )

        # 加入辅助训练模块的loss,这可以让网络在初期收敛的更快
        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                #----------------------------------------------------------------------------------------
                # 在字典中加入aux_loss的键值,方便在稍后的bp中选择是否让aux head进行梯度会回传
                loss_states["aux_" + k] = v
        return loss, loss_states




    # 根据标签分配结果计算loss
    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign

        # 因为要对整个batch进行平均,因此,在这里计算出这次分配的总正样本数,用于稍后的weight_loss
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        # 为了一次性处理一个batch的数据把每个结果都拼接起来
        # labels和label_score都是[batchsize*2125],bbox_targets是[batchsize*2125,4]
        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)

        # 把预测结果和检测框都reshape成和batch对应的形状
        cls_preds = cls_preds.reshape(-1, self.num_classes) # [batchsize*2125,class_num]
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1)) # [batchsize*2125,4*(reg_max+1)]
        decoded_bboxes = decoded_bboxes.reshape(-1, 4) # [batchsize*2125,4]

        # 计算quality focal loss,参见GFL论文,笔者也有一篇文章讲解GFL.
        # 利用IOU联合了框的质量估计和分类表示,和软标签计算相似
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )
        
        # 获取预测对应的lable标签(把当前batch的所有index交给pos_inds)
        # tensor中常用逻辑判断语句生成mask掩膜,元素中符合者变成True,反之为False
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        # 如果label为空说明没有分配任何标,那就不用计算dfl和iou loss了
        if len(pos_inds) > 0:
            # 计算用于weight_reduce的参数,weight_target的长度和被分配了gt的prior的数量相同
            # sigmoid后取最大值得到的就是该prior输出的类别分数
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            # 同步GPU上的其他worker,获得此参数
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            # 计算GIoU损失,加入了weighted_loss
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            # 同样拼接起来方便批量计算
            dist_targets = torch.cat(dist_targets, dim=0)
            # 计算Distribution focal loss
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states



    # 标签分配时的运算不会被记录,我们只是在计算cost并进行匹配
    # 需要特别注意,这个函数只为一张图片,即一个样本进行标签分配!
    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        这些参数和第四部分介绍的label assigner差不多,稍后也会调用assign
        主要是为能够进行批处理而增加了一些代码
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        device = center_priors.device
        # 一些前处理,把数据都移到gpu里面
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        # dist_targets是最终用来计算回归损失的tensor,具体过程看后面
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)

        # 把label扩充成one-hot向量
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)

        # No target,啥也不用管,全都是负样本,返回回去就行
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0

        # class的输出要映射到0-1之间,看之前对head构建conv layer就可以发现最后的分类输出后面没带激活函数
        # assign参见第四部分关于dsl_assigner的介绍
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        # 调用采样函数,获得正负样本
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        # 当前进行分配的这个图片上正样本的数目
        num_pos_per_img = pos_inds.size(0)
        # 把分配到了gt的那些prior预测的检测框和gt的iou算出来,稍后用于QFL的计算
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            # bbox_targets就是最终用来和gt计算回归损失的东西了(reg分支),维度为[2125,4]
            # 不过这里还需要一步转化,因为bbox_target是检测框四条边和他对应的prior的偏移量
            # 因此要转换成原图上的框(绝对位置),和gt进行回归损失计算
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            # 
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            # 上面计算回归,这里就是得到用于计算类别损失的,把那些匹配到的prior利用pos_inds索引筛出来
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )




    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        # 分配到标签的priors索引,注意正样本和负样本的大小总和应该为prior的数目
        # 对于320x320,是2125
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        # 没分配到标签的priors索引
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        # -----------------------------------------------------------------------
        # @TODO:
        # 这里有疑问,不知道为什么之前在dsl_assigner里面要将分配到标签的prior索引变成2
        # 其实不是直接设置成1就可以了吗?然后这里又重新把index-1,有点不太明白
        # 如果有看懂了的或者知道为什么,请联系我!
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        #------------------------------------------------------------------------

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            # pos_gt_bboxes大小为[正样本数,4],4就是框的位置了
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds





    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results





    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result





    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list





    # 在feature map上布置一组prior
    # prior就是框分布的回归起点,将以prior的位置作为目标中心,预测四个值形成检测框
    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        # arange()会生成一个一维tensor,和range()差不多,步长默认为1
        # 乘上stride后,就得到了划分的网格宽度和长度
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        # 根据网格长宽生成一组二维坐标
        y, x = torch.meshgrid(y_range, x_range)
        # 展平成一维
        y = y.flatten()
        x = x.flatten()
        # 扩充出一个strides的tensor,稍后给每一个prior都加上其对应的下采样倍数
        strides = x.new_full((x.shape[0],), stride)
        # 把得到的prior按照以下顺序叠成二维tensor,即原图上的坐标,采样倍数
        proiors = torch.stack([x, y, strides, strides], dim=-1)

        # 一次处理一个batch,所以unsqueeze增加一个batch的维度
        # 然后把得到的prior复制到同个batch的其他位置上
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)





    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)
