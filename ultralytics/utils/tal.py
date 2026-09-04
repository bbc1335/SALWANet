# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import re

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where
                topk is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos
    

class NewTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0):
        super().__init__(topk, num_classes, alpha, beta)
        # self.S0 = 128
        self.S0 = 550.0  # VisDrone
        # self.S0 = 256.0  # AIToD
        self.k = 0.008  # steepness of sigmoid for Elastic Center Prior (ECP)
        # self.k = 0.005  # steepness of sigmoid for Elastic Center Prior (ECP)
        self.topk_low = 3 # minimum fusion weight for NPD
        # self.topk_high = 6 # maximum fusion weight for NPD
        self.topk_high = 10 # maximum fusion weight for NPD

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps, gt_areas = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes, gt_areas)
        # target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
        
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        expanded_gt_bboxes = self.expand_gt_boxes_pixel(gt_bboxes, S0=self.S0, k=self.k)
        mask_in_gts = self.select_candidates_in_gts(anc_points, expanded_gt_bboxes)
        # mask_in_gts1 = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        # mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        dynamic_k, gt_areas = self.get_dynamic_topk(gt_bboxes, mask_gt)
        mask_topk = self.select_topk_candidates_dynamic(align_metric, dynamic_k, mask_gt)
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps, gt_areas

    def get_dynamic_topk(self, gt_bboxes, mask_gt):
        """根据 GT 面积计算动态 topk 值，面积越小 topk 越大"""
        gt_areas = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        # 线性映射：面积 0 → topk_high，面积 max_area → topk_low
        ratio = torch.clamp(gt_areas / self.S0, 0.0, 1.0)
        dynamic_k = self.topk_high - ratio * (self.topk_high - self.topk_low)
        dynamic_k = dynamic_k.long() * mask_gt.squeeze(-1)  # 无效 GT 置 0
        return dynamic_k, gt_areas  # [bs, n_max_boxes]

    def select_topk_candidates_dynamic(self, metrics, k_values, topk_mask=None):
        """
        在原 select_topk_candidates 基础上改造，支持每个 GT 使用不同的 topk 值。

        Args:
            metrics (torch.Tensor): (b, max_num_obj, h*w) 对齐度量分数。
            k_values (torch.Tensor): (b, max_num_obj) 整数张量，每个 GT 的动态 topk 值。
            topk_mask (torch.Tensor, optional): 形状 (b, max_num_obj, max_k) 的布尔掩码，用于过滤无效候选。

        Returns:
            (torch.Tensor): (b, max_num_obj, h*w) 的 0/1 掩码，表示被选中的候选锚点。
        """
        # 确定本次计算的最大 k 值（用于确定取多少个 topk 索引）
        max_k = int(k_values.max().item())
        if max_k == 0:
            return torch.zeros_like(metrics, dtype=metrics.dtype)

        # 1. 取出全局最大 k 个候选索引（多取一些，后面根据各 GT 的 k 值截断）
        #    topk_idxs: (b, max_num_obj, max_k)
        topk_metrics, topk_idxs = torch.topk(metrics, max_k, dim=-1, largest=True)

        # 2. 处理 topk_mask（若未提供则根据得分阈值自动生成）
        if topk_mask is None:
            # 原逻辑：保留那些最大得分 > eps 的 GT
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # 确保 topk_mask 形状与 topk_idxs 一致 (b, max_num_obj, max_k)
        topk_idxs.masked_fill_(~(topk_mask.expand(-1, -1, max_k).bool()), 0)

        # 3. 初始化计数张量
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)

        # 4. 逐 k 循环（最多循环 max_k 次）
        for k in range(max_k):
            # 构造当前 k 位置的有效性掩码：只有 k < 该 GT 的 k_values 时才执行 scatter_add
            # valid_mask: (b, max_num_obj, 1) 布尔型，True 表示该 GT 的 topk 包含第 k 个候选
            valid_mask = (k < k_values).unsqueeze(-1)  # (b, max_num_obj, 1)

            # 将 valid_mask 与当前 k 对应的 topk_idxs 切片结合
            # 只对有效位置进行 scatter_add
            idx_slice = topk_idxs[:, :, k:k+1]  # (b, max_num_obj, 1)

            # 使用 masked_scatter 或逐元素操作：这里采用 torch.where 生成带条件的 ones
            # 当 valid_mask 为 False 时，将 ones 置零，避免错误累加
            masked_ones = ones * valid_mask.to(ones.dtype)

            # 累加到计数张量
            count_tensor.scatter_add_(-1, idx_slice, masked_ones)

        # 5. 过滤无效边界框（若某个锚点被同一 GT 多次选中，置 0）
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)
    
    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes, gt_areas):
        """
        尺度感知冲突解决：优先将锚点分配给面积较小的 GT。

        Args:
            mask_pos: (b, n_max_boxes, na)
            overlaps: (b, n_max_boxes, na) CIoU
            n_max_boxes: int
            gt_areas: (b, n_max_boxes) GT 面积
            scale_alpha: 尺度优先强度，0 表示退化为原版
        """
        fg_mask = mask_pos.sum(-2)  # (b, na)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)

            # 计算尺度权重：面积越小，权重越高
            # 对面积做归一化，防止量纲影响
            norm_areas = gt_areas / (gt_areas.max(dim=1, keepdim=True)[0] + 1e-6)  # (b, n_max_boxes)
            scale_weight = 1.0 / (norm_areas.unsqueeze(-1) + 1e-6) ** 0.5   # (b, n_max_boxes, 1)

            weighted_overlaps = overlaps * scale_weight

            max_idx = weighted_overlaps.argmax(1)  # (b, na)

            is_max = torch.zeros_like(mask_pos)
            is_max.scatter_(1, max_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max, mask_pos)
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos
    
    # def select_topk_candidates_dynamic(self, metrics, gt_bboxes, mask_gt):
    #     """
    #     对每个 GT 独立计算动态 k 值，并选择 Top-K。
    #     Args:
    #         metrics: (bs, n_gt, na)  对齐度量
    #         gt_bboxes: (bs, n_gt, 4) [x1,y1,x2,y2]
    #         mask_gt: (bs, n_gt, 1)   有效GT掩码
    #     Returns:
    #         mask_topk: (bs, n_gt, na) 二值掩码，1表示被选为正样本
    #     """
    #     bs, n_gt, na = metrics.shape
    #     device = metrics.device

    #     # 计算每个 GT 的面积 (像素)
    #     w = gt_bboxes[..., 2] - gt_bboxes[..., 0]   # (bs, n_gt)
    #     h = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    #     area_gt = w * h + self.eps                   # 避免除零

    #     # 动态 k 值: k = base * (area_ref / area)^gamma
    #     ratio = self.S0 / area_gt
    #     k_dynamic = self.topk * (ratio ** 0.4)
    #     k_dynamic = torch.clamp(k_dynamic, self.k_min, self.k_max).long()  # (bs, n_gt)
    #     max_k = k_dynamic.max().item()

    #     # 对所有 GT 取 Top-max_k，得到指标和索引
    #     topk_metrics, topk_idxs = torch.topk(metrics, max_k, dim=-1, largest=True)  # (bs, n_gt, max_k)
    #     if mask_gt is None:
    #         mask_gt = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

    #     # 生成有效掩码：每个 GT 只有前 k_dynamic 个有效
    #     k_mask = torch.arange(max_k, device=device).view(1, 1, max_k) < k_dynamic.unsqueeze(-1)  # (bs, n_gt, max_k)
    #     topk_idxs = topk_idxs.masked_fill(~k_mask, 0)

    #     # 构建 count_tensor (与原始 TAL 一致)
    #     count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=device)
    #     ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
    #     for k_idx in range(max_k):
    #         count_tensor.scatter_add_(-1, topk_idxs[:, :, k_idx:k_idx+1], ones)
    #     # 过滤掉重复分配的 anchor (一个 anchor 被多个 GT 选中)
    #     count_tensor.masked_fill_(count_tensor > 1, 0)

    #     return count_tensor.to(metrics.dtype)
    
    def expand_gt_boxes_pixel(self, gt_bboxes, delta_min=0.05, delta_max=0.30, S0=1024, k=None):
        """
        Elastic Center Prior (ECP): Expand ground truth boxes based on absolute pixel area.

        This function performs scale-adaptive expansion of GT boxes to increase candidate
        anchor points for small objects, without requiring explicit image dimensions.
        Clipping to image boundaries is omitted as out-of-bounds expansion does not
        introduce any valid anchor points (anchors always reside within the image).

        Args:
            gt_bboxes (Tensor): shape (bs, n_max, 4) in pixel coordinates (x1, y1, x2, y2).
            delta_min (float): minimum expansion ratio (for large objects).
            delta_max (float): maximum expansion ratio (for extremely small objects).
            S0 (float): area threshold (pixels²), COCO small object boundary (32x32 = 1024).
            k (float, optional): steepness of the sigmoid transition. If None, defaults to 2/S0.

        Returns:
            Tensor: expanded gt_bboxes, same shape and device as input.
        """
        if k is None:
            k = 2.0 / S0  # smooth transition over [0, 2*S0]

        w = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        area = w * h  # (bs, n_max), in pixels²

        # Sigmoid expansion factor (Equation 1)
        delta = delta_min + (delta_max - delta_min) / (1.0 + torch.exp(k * (area - S0)))
        # Clamp to valid range for numerical stability
        delta = torch.clamp(delta, delta_min, delta_max)

        dw = delta * w
        dh = delta * h

        expanded = gt_bboxes.clone()
        expanded[..., 0] = gt_bboxes[..., 0] - dw
        expanded[..., 1] = gt_bboxes[..., 1] - dh
        expanded[..., 2] = gt_bboxes[..., 2] + dw
        expanded[..., 3] = gt_bboxes[..., 3] + dh

        # Note: no explicit clipping to image boundaries is performed.
        # Expanded regions outside the image will never contain any anchor points,
        # thus they do not affect the final assignment result.

        return expanded

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
    
    def npd_similarity(self, gt_boxes, pd_boxes, eps=1e-7):
        """
        Compute Normalized Projection Distance similarity between two sets of boxes.
        Boxes are in (x1, y1, x2, y2) format.
        
        Args:
            gt_boxes (Tensor): shape (..., 4)
            pd_boxes (Tensor): shape (..., 4)
            eps (float): small constant for numerical stability.
        
        Returns:
            Tensor: NPD similarity, shape gt_boxes.shape[:-1]
        """
        # Intersection limits
        inter_x1 = torch.max(gt_boxes[..., 0], pd_boxes[..., 0])
        inter_y1 = torch.max(gt_boxes[..., 1], pd_boxes[..., 1])
        inter_x2 = torch.min(gt_boxes[..., 2], pd_boxes[..., 2])
        inter_y2 = torch.min(gt_boxes[..., 3], pd_boxes[..., 3])

        # Outer bounding box limits (union of extents)
        outer_x1 = torch.min(gt_boxes[..., 0], pd_boxes[..., 0])
        outer_y1 = torch.min(gt_boxes[..., 1], pd_boxes[..., 1])
        outer_x2 = torch.max(gt_boxes[..., 2], pd_boxes[..., 2])
        outer_y2 = torch.max(gt_boxes[..., 3], pd_boxes[..., 3])

        # Widths and heights
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        outer_w = (outer_x2 - outer_x1).clamp(min=eps)
        outer_h = (outer_y2 - outer_y1).clamp(min=eps)

        overlap_x = inter_w / outer_w
        overlap_y = inter_h / outer_h

        npd = overlap_x * overlap_y

        # # 引入中心距离约束
        # gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        # gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        # pd_center_x = (pd_boxes[..., 0] + pd_boxes[..., 2]) / 2
        # pd_center_y = (pd_boxes[..., 1] + pd_boxes[..., 3]) / 2
        # dist_center = torch.sqrt((gt_center_x - pd_center_x)**2 + (gt_center_y - pd_center_y)**2)
        # max_size = torch.max(outer_w, outer_h)
        # center_penalty = torch.exp(-dist_center**2 / (2 * max_size**2))  # 高斯惩罚

        return npd
    
    def nwd_similarity(self, 
                       gt_boxes, 
                       pd_boxes, 
                       ):
        """
        动态归一化的 NWD 相似度。
        gt_boxes, pd_boxes: (..., 4) [x1,y1,x2,y2]
        返回: (..., ) 相似度，范围 (0,1]
        """
        # 转换为中心宽高
        gt_cx = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_cy = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        gt_w = (gt_boxes[..., 2] - gt_boxes[..., 0]).clamp(min=self.eps)
        gt_h = (gt_boxes[..., 3] - gt_boxes[..., 1]).clamp(min=self.eps)

        pd_cx = (pd_boxes[..., 0] + pd_boxes[..., 2]) / 2
        pd_cy = (pd_boxes[..., 1] + pd_boxes[..., 3]) / 2
        pd_w = (pd_boxes[..., 2] - pd_boxes[..., 0]).clamp(min=self.eps)
        pd_h = (pd_boxes[..., 3] - pd_boxes[..., 1]).clamp(min=self.eps)

        # 中心距离平方
        center_dist2 = (gt_cx - pd_cx) ** 2 + (gt_cy - pd_cy) ** 2
        # 半宽高差平方
        w_half_diff = (gt_w - pd_w) / 2
        h_half_diff = (gt_h - pd_h) / 2
        w2_sq = center_dist2 + w_half_diff ** 2 + h_half_diff ** 2
        w2 = torch.sqrt(w2_sq + self.eps)   # Wasserstein 距离

        # 动态归一化：以 GT 框的半对角线长度为尺度
        gt_scale = torch.sqrt(gt_w * gt_h)
        norm_scale = self.alpha_nwd * gt_scale.clamp(min=self.eps)
        sim = torch.exp(-w2 / (norm_scale + self.eps))
        return sim


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
