import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

'''
    dynamic soft label assigner,根据pred和GT的IOU进行软标签分配
    某个pred与GT的IOU越大,最终分配给它的标签值会越接近一,反之会变小
'''

class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
        iou_factor (float): The scale factor of iou cost. Default 3.0.
    """
    def __init__(self, topk=13, iou_factor=3.0):
        self.topk = topk
        self.iou_factor = iou_factor

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
    ):
        """Assign gt to priors with dynamic soft label assignment.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, cy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        # 切片得到prior位置(可以看作anchor point的中心点)
        prior_center = priors[:, :2]
        # 计算prior center到GT左上角和右下角的距离,从而判断prior是否在GT框内
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        # is_in_gts通过判断deltas全部大于零筛选处在gt中的prior
        # [dxlt,dylt,dxrb,dyrb]四个值都需要大于零,则它们中最小的值也要大于零
        # tensor.min会返回一个 namedtuple (values, indices),-1代表最后一个维度
        # 其中 values 是给定维度 dim 中输入张量的每一行的最小值,并且索引是找到的每个最小值的索引位置
        # 这里判断赋值bool,若prior落在gt中对应的[prior_i,gt_j]会变为true
        # [i,j]代表第i个prior是否落在第j个ground truth中
        is_in_gts = deltas.min(dim=-1).values > 0
        # 这一步生成有效prior的索引,这里请注意之所以用sum是因为一个prior可能落在多个GT中
        # 因此上一步生成的is_in_gts确定的是某个prior是否落在每一个GT中,只要落在一个GT范围内,便是有效的
        valid_mask = is_in_gts.sum(dim=1) > 0

        # 利用得到的mask确定由哪些prior生成的pred_box和它们对应的scores是有效的,注意它们的长度
        # 注意valid_decoded_bbox和valid_pred_scores的长度是落在gt中prior的个数
        # 稍后在dynamic_k_matching()我们会再提到这一点
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        # 出现没有预测框或者训练样本中没有GT的情况
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                # 通过这种方式,可以直接在数据集中放置没有标签的图像作为负样本
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1, dtype=torch.long
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # 计算有效bbox和gt的iou损失
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        # clamp,加上一个很小的数防止出现NaN
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        # 根据num_valid的数量(有效bbox)生成对应长度的one-hot label之后用于计算soft lable
        # 每个匹配到gt的prior都会有一个[0,0,...,1,0]的tensor,即label位置的元素为一其余为零
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        # IOU*onehot得到软标签,直觉上非常好理解,预测框和gt越接近,说明预测的越好
        # 那么,稍后计算交叉熵的时候,标签值也会更大
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        # 计算缩放权重因子,为软标签减去该prior预测的bbox的score
        # 可以想象,差距越大说明当前的预测效果越差,稍后的cost计算就应该给一个更大的惩罚
        scale_factor = soft_label - valid_pred_scores

        # 计算分类交叉熵损失
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)

        # 最后得到的匹配开销矩阵,数值为分类损失和IOU损失,这里利用iou_factor作为调制系数
        cost_matrix = cls_cost + iou_cost * self.iou_factor

        # 返回值为分配到标签的prior与它们对应的gt的iou和这些prior匹配到的gt索引
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        # 把结果还原为priors的长度
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        """
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        # pairwise_ious匹配成功的组数可能会小于默认的topk,这里取两者最小值防止越界
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        # 从IOU矩阵中选出IOU最大的topk个匹配
        # topk函数返回一个namedtuple(value,indices),indices没用,python里无用变量一般约定用"_"接收
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        # 用topk个预测IOU值之和作为一个GT要分配给prior的个数
        # 这个想法很直观,可以把IOU为1看作一个完整目标,那么这些预测框和GT的IOU总和就是最终分配的个数
        # clamp规约,最小值为一,因为不可能一个都不给分配,dynmaic_ks的大小和GT个数相同
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        # 对每一个GT,挑选出上面计算出的dymamic_ks个拥有最小cost的预测
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False # False则返回最小值
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0  # 被匹配的(prior,gt)位置上被置1

        del topk_ious, dynamic_ks, pos_idx

        # 第二个维度是prior的维度，大于1说明一个prior匹配到了多个GT,这里要选出匹配cost最小的GT
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            # 匹配到多个GT的prior的行全部清零
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            # 把这些prior和gt有最小cost的位置置1
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        # 统计matching_matrix中被分配了标签的prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0

        '''
        假设priors长度是n,并且有m个prior落在gt中,那么valid_mask长度也是n,并且有m个true,n-m个false;
        在这m个落在gt中的prior里,又有k个被匹配到了,故fg_mask_inboxes的维度是m,其中有k个位置为true;
        因此valid_mask中有m-k个位置的true也需要被置为false.
        在这里valid_mask[valid_mask]会把原来所有为true的位置索引返回,让他们等于fg_mask_inboxes
        维度对照表:
            n为priors长度即所有prior个数,m为落在gt中的prior个数,k为匹配到gt的prior的个数,g为gt个数
            valid_mask:[n]          其中m个为true
            matching_matrix:[gt,m]  每一列至多只有一个为1
            fg_mask_inboxes:[m]     其中k个为true
        最终得到的valid_mask中只有k个为true剩余为false
        '''
        # 注意这个索引方式,valid_mask是一个bool类型tensor,以自己为索引会返回所有为True的位置
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        # 找到已被分配标签的prior对应的gt index,argmax返回最大值所在的索引,每一个prior只会对应一个GT
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # 同上,把它们的IOU提取出来
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
