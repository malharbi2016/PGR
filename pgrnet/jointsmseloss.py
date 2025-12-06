# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Wei Yang (platero.yang@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F


class JointsCombinedLoss(nn.Module):
    """
    L_total = L_loc + λ1 * L_pc + λ2 * L_sk

    - L_loc : your original joint heatmap MSE loss
    - L_pc  : prompt consistency loss  ||P - P_gt_bar||_2^2
    - L_sk  : skeleton structural loss over centroids
              sum_i ||(c_{i+1} - c_i) - (c^gt_{i+1} - c^gt_i)||_2^2

    The forward signature is kept compatible with your old loss:
        forward(output, target, target_weight, ...)

    """

    def __init__(self,
                 use_target_weight: bool = True,
                 lambda_pc: float = 0.0,
                 lambda_sk: float = 0.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.lambda_pc = lambda_pc
        self.lambda_sk = lambda_sk
        self._localization_loss = JointsMSELoss(use_target_weight=use_target_weight)

        # base MSE
        self.criterion = nn.MSELoss(reduction='mean')

    def _prompt_consistency_loss(self, prompt, prompt_gt):
        """
        L_pc = ||P - P_gt_bar||_2^2
        prompt, prompt_gt: [B, D] or [B, C, H, W] – any same shape tensor.
        """
        if prompt is None or prompt_gt is None:
            return 0.0

        return self.criterion(prompt, prompt_gt)

    def _skeleton_structural_loss(self, output, target):
        """
        Skeleton structural loss L_sk.

        output, target: [B, V, H, W] Gaussian heatmaps
            - B: batch size
            - V: number of discs / joints
            - H, W: heatmap size

        Computes centroids via soft-argmax (center of mass) for each joint,
        then applies:
            L_sk = mean_i || (ĉ_{i+1} - ĉ_i) - (c^gt_{i+1} - c^gt_i) ||_2^2
        """
        B, V, H, W = output.shape
        device = output.device

        # Flatten spatially: [B, V, H*W]
        pred = output.reshape(B, V, -1)
        gt   = target.reshape(B, V, -1)

        # Ensure non-negative and normalize to get spatial probabilities
        pred = torch.relu(pred)
        gt   = torch.relu(gt)

        pred_sum = pred.sum(dim=-1, keepdim=True) + 1e-6
        gt_sum   = gt.sum(dim=-1, keepdim=True) + 1e-6

        pred = pred / pred_sum
        gt   = gt / gt_sum

        # Coordinate grid (x: 0..W-1, y: 0..H-1)
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        xs = xs.reshape(-1).float()  # [H*W]
        ys = ys.reshape(-1).float()  # [H*W]

        # Expected coordinates (soft-argmax)
        cx_pred = (pred * xs).sum(dim=-1)  # [B, V]
        cy_pred = (pred * ys).sum(dim=-1)  # [B, V]
        cx_gt   = (gt   * xs).sum(dim=-1)  # [B, V]
        cy_gt   = (gt   * ys).sum(dim=-1)  # [B, V]

        # Stack to get centroids: [B, V, 2]
        centroids_pred = torch.stack([cx_pred, cy_pred], dim=-1)
        centroids_gt   = torch.stack([cx_gt,   cy_gt],   dim=-1)

        # Pairwise differences along the column: [B, V-1, 2]
        diff_pred = centroids_pred[:, 1:, :] - centroids_pred[:, :-1, :]
        diff_gt   = centroids_gt[:,   1:, :] - centroids_gt[:,   :-1, :]

        # L_sk = mean squared L2 difference
        loss_sk = torch.mean((diff_pred - diff_gt) ** 2)

        return loss_sk

    def forward(self,
                output,
                target,
                target_weight,
                prompt = None,
                prompt_gt = None):
        """
        Keeps the original call:
            loss = criterion(output, target, target_weight)

        If you want the extra terms, call:
            loss = criterion(output, target, target_weight,
                             prompt=P,
                             prompt_gt=P_gt_bar,
                             centroids_pred=c_hat,
                             centroids_gt=c_gt)
        """
        # 1) localization loss (heatmaps)
        loc_loss = self._localization_loss(output, target, target_weight)

        # 2) prompt consistency loss
        pc_loss = self._prompt_consistency_loss(prompt, prompt_gt)

        # 3) skeleton structural loss
        sk_loss = self._skeleton_structural_loss(output, target)

        total_loss = loc_loss + self.lambda_pc * pc_loss + self.lambda_sk * sk_loss

        return total_loss




class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
