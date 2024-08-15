import os
import logging
import transformers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import ListConfig
from typing import List, Union
from yarr.agents.agent import Agent, ActResult, ScalarSummary, Summary

from voxel.augmentation import apply_se3_augmentation
from helpers.clip.core.clip import build_model, load_clip
from helpers.optim.lamb import Lamb

NAME = 'SGR_Agent'
EPSILON = 1e-7


class SGRNetWrapper(nn.Module):

    def __init__(self, sgr_network: nn.Module, trans_num_classes_per_axis: int,
                 rotation_resolution: float, device, training):
        super(SGRNetWrapper, self).__init__()
        self._trans_num_classes_per_axis = trans_num_classes_per_axis
        self._rotation_resolution = rotation_resolution
        self._sgr_net = sgr_network.to(device)

        # distributed training
        if training:
            # sync_bn
            self._sgr_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self._sgr_net)
            self._sgr_net = DDP(self._sgr_net,
                                device_ids=[device],
                                find_unused_parameters=True)

    def choose_highest_action(self,
                              q_trans,
                              q_rot_grip,
                              q_collision,
                              rot_cls=True):

        if q_trans is not None:
            q_trans_flat = torch.stack(torch.split(
                q_trans, int(self._trans_num_classes_per_axis), dim=1),
                                       dim=1)
            coords = torch.cat([
                q_trans_flat[:, 0:1].argmax(-1),
                q_trans_flat[:, 1:2].argmax(-1), 
                q_trans_flat[:, 2:3].argmax(-1),
            ], -1)
        else:
            coords = None

        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            if rot_cls:
                q_rot = torch.stack(torch.split(q_rot_grip[:, :-2],
                                                int(360 // self._rotation_resolution),
                                                dim=1),
                                    dim=1)  # (B, 3, 72)
                rot_and_grip_indicies = torch.cat([
                    q_rot[:, 0:1].argmax(-1), q_rot[:, 1:2].argmax(-1),
                    q_rot[:, 2:3].argmax(-1), q_rot_grip[:, -2:].argmax(
                        -1, keepdim=True)
                ], -1)
            else:
                rot_and_grip_indicies = torch.cat([
                    q_rot_grip[:, :-2], q_rot_grip[:, -2:].argmax(-1, keepdim=True)
                ], -1)  # (B, 4 + 1), quat + grip indicies

            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                rgb_pcd,
                proprio,
                pcd,
                lang_goal_emb,
                lang_token_embs,
                bounds=None,
                feat_mask=None,
                pcd_bound_masks=None):

        b = rgb_pcd[0][0].shape[0]
        rgb = [rp[0] for rp in rgb_pcd]

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        q_trans, q_rot_and_grip, q_ignore_collisions, pred_dict = self._sgr_net(
            pcd,
            rgb,
            bounds,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            feat_mask,
            pcd_bound_masks=pcd_bound_masks)

        return q_trans, q_rot_and_grip, q_ignore_collisions, pred_dict


class SGRAgent(Agent):

    def __init__(
        self,
        layer: int,
        coordinate_bounds: list,
        sgr_network: nn.Module,
        camera_names: list,
        batch_size: int,
        voxel_size: int,
        num_rotation_classes: int,
        rotation_resolution: float,
        include_low_dim_state: bool = False,
        lr: float = 0.0001,
        optimizer_type: str = 'adamw',
        lr_scheduler: bool = False,
        scheduler_type: str = 'cosine',
        training_iterations: int = 100000,
        num_warmup_steps: int = 20000,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
        transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
        transform_augmentation_rot_resolution: int = 5,
        bound_pcd_before_transform: bool = False,
        trans_cls: bool = False,
        rot_cls: bool = True,
        regression_loss: str = 'l2',
        color_drop: float = 0.0,
        feat_drop: float = 0.0,
        trans_point_loss: bool = False,
        rot_point_loss: bool = False,
        temperature: Union[float, List[float]] = 0.1,
    ):
        self._layer = layer
        self._coordinate_bounds_list = coordinate_bounds
        self._sgr_network = sgr_network
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._include_low_dim_state = include_low_dim_state

        self._lr = lr
        self._optimizer_type = optimizer_type
        self._lr_scheduler = lr_scheduler
        self._scheduler_type = scheduler_type
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._bound_pcd_before_transform = bound_pcd_before_transform

        self._trans_cls = trans_cls
        self._rot_cls = rot_cls
        self._regression_loss = regression_loss

        self._color_drop = color_drop
        self._feat_drop = feat_drop
        self._trans_point_loss = trans_point_loss
        self._rot_point_loss = rot_point_loss

        if isinstance(temperature, list) or isinstance(temperature,
                                                       ListConfig):
            self._temperature_trans = temperature[0]
            self._temperature_rot_grip_collision = temperature[1]
        elif isinstance(temperature, float):
            self._temperature_trans = temperature
            self._temperature_rot_grip_collision = temperature
        else:
            raise ValueError

        self._mse_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        self._net = SGRNetWrapper(self._sgr_network, self._voxel_size,
                                  self._rotation_resolution, device,
                                  training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds_list,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._net.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._net.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            elif self._optimizer_type == 'adamw':
                self._optimizer = torch.optim.AdamW(
                    self._net.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                if self._scheduler_type == 'cosine_with_hard_restarts':
                    self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                        self._optimizer,
                        num_warmup_steps=self._num_warmup_steps,
                        num_training_steps=self._training_iterations,
                        num_cycles=self._training_iterations // 10000,
                    )
                elif self._scheduler_type == 'cosine':
                    self._scheduler = transformers.get_cosine_schedule_with_warmup(
                        self._optimizer,
                        num_warmup_steps=self._num_warmup_steps,
                        num_training_steps=self._training_iterations,
                    )
                else:
                    raise Exception('Unknown scheduler type')

            # one-hot zero tensors
            if self._trans_cls:
                self._action_trans_x_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._voxel_size),
                    dtype=int,
                    device=device)
                self._action_trans_y_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._voxel_size),
                    dtype=int,
                    device=device)
                self._action_trans_z_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._voxel_size),
                    dtype=int,
                    device=device)
            if self._rot_cls:
                self._action_rot_x_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._num_rotation_classes),
                    dtype=int,
                    device=device)
                self._action_rot_y_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._num_rotation_classes),
                    dtype=int,
                    device=device)
                self._action_rot_z_one_hot_zeros = torch.zeros(
                    (self._batch_size, self._num_rotation_classes),
                    dtype=int,
                    device=device)
            self._action_grip_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device)

        else:
            for param in self._net.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._net.to(device)

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _mseloss(self, pred, labels):
        return self._mse_loss(pred, labels)

    def _l1loss(self, pred, labels):
        return self._l1_loss(pred, labels)

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _pointwise_celoss(self, pred, labels):
        # Input: pred(logits): (B, N, C), labels: (B, N, C)
        # Output: loss: (B, N)
        labels = labels.argmax(-1)
        B, N, C = pred.shape
        losses = F.cross_entropy(pred.view(-1, C),
                                 labels.view(-1),
                                 reduction='none')
        pointwise_loss = losses.view(B, N)
        return pointwise_loss

    def _pointwise_diff(self, pred, labels):
        # Input: pred(logits or softmax): (B, N, C) / (B, C), labels: (B, N, C) / (B, C)
        # Output: pointwise_diff: (B, N) / (B,)
        labels = labels.argmax(-1)
        pred = pred.argmax(-1)
        pointwise_diff = (torch.abs(pred - labels)).float()
        return pointwise_diff

    def _celoss_softmax(self, probs, labels):
        # Input: probs: (B, C), labels: (B, C)
        # Output: loss: scalar
        labels = labels.argmax(-1)
        probs = torch.clamp(probs, min=EPSILON)
        actual_probs = probs.gather(1, labels.unsqueeze(-1))
        losses = -torch.log(actual_probs.squeeze())
        average_loss = losses.mean()
        return average_loss

    def _quat_loss(self, pred, labels):
        # shape: (b, 4)
        dot_product = torch.sum(pred * labels, dim=1)
        loss = 1 - dot_product
        return loss.mean()

    def _softmax_q_trans(self, q):
        # q_shape = q.shape
        # return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)
        q_trans_x_flat = q[:, 0 * self._voxel_size:1 * self._voxel_size]
        q_trans_y_flat = q[:, 1 * self._voxel_size:2 * self._voxel_size]
        q_trans_z_flat = q[:, 2 * self._voxel_size:3 * self._voxel_size]

        q_trans_x_flat_softmax = F.softmax(q_trans_x_flat, dim=1)
        q_trans_y_flat_softmax = F.softmax(q_trans_y_flat, dim=1)
        q_trans_z_flat_softmax = F.softmax(q_trans_z_flat, dim=1)

        return torch.cat([
            q_trans_x_flat_softmax, q_trans_y_flat_softmax,
            q_trans_z_flat_softmax
        ],
                         dim=1)

    def _softmax_q_rot_grip(self, q_rot_grip):
        if self._rot_cls:
            q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 *
                                      self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 *
                                      self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 *
                                      self._num_rotation_classes]
        else:
            q_rot_quat = q_rot_grip[:, :4]

        q_grip_flat = q_rot_grip[:, -2:]

        if self._rot_cls:
            q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
            q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
            q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        if self._rot_cls:
            out = torch.cat([
                q_rot_x_flat_softmax, q_rot_y_flat_softmax,
                q_rot_z_flat_softmax, q_grip_flat_softmax
            ], dim=1)
        else:
            out = torch.cat([q_rot_quat, q_grip_flat_softmax], dim=1)

        return out

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def get_pcd_bound_masks(self, pcd, bounds):
        # pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
        # bounds: [1, 6]
        pcd_bound_masks = []

        for pc in pcd:
            x_min, y_min, z_min, x_max, y_max, z_max = bounds[0]
            in_x = (pc[:, 0, :, :] >= x_min) & (pc[:, 0, :, :] <= x_max)
            in_y = (pc[:, 1, :, :] >= y_min) & (pc[:, 1, :, :] <= y_max)
            in_z = (pc[:, 2, :, :] >= z_min) & (pc[:, 2, :, :] <= z_max)

            mask = (in_x & in_y & in_z).int()  # [bs, H, W]
            pcd_bound_masks.append(mask)

        return pcd_bound_masks

    def update(self, step: int, replay_sample: dict) -> dict:
        # What other agent do on update function:
        # preprocess_agent.py: normlize rgb to [-1,1]; (only use fisrt sample from buffer, i.e. t=1)
        # stack_agent.py: (on act) output continuous_action, discrete_euler_to_quaternion

        # Input: replay_sample (dict)
        # targer_action: trans, rot, grip, ignore_collision [B, ]
        # language: lang_goal_emb [B, ]
        # proprio = replay_sample['low_dim_state'] [B, Cp] (Cp=4 in RLBench)
        # observation(before self._preprocess_inputs):  replay_sample['%s_rgb'%cam_name]  
        # replay_sample['%s_point_cloud'%cam_name] [B, 3, H, W]
        # observation(after self._preprocess_inputs): obs: [[rgb1, pcd1], [rgb2, pcd2],... ]; pcd: [pcd1, pcd2, ...]

        # (1) obtain input
        action_trans = replay_sample['trans_action_indicies'][
            :, self._layer * 3:self._layer * 3 + 3].int()
        action_trans_continuous = replay_sample['trans_action_continuous'][
            :, self._layer * 3:self._layer * 3 + 3].float()  # if _transform_augmentation, this will not be used
        action_rot_grip = replay_sample['rot_grip_action_indicies']
        quat = replay_sample['quat'].float()
        action_gripper_pose = replay_sample['gripper_pose']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        #language
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        device = self._device

        bounds = self._coordinate_bounds.to(device)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        obs, pcd = self._preprocess_inputs(replay_sample)

        # batch size
        bs = pcd[0].shape[0]

        if self._bound_pcd_before_transform:
            pcd_bound_masks = self.get_pcd_bound_masks(pcd, bounds)
        else:
            pcd_bound_masks = None

        # (2) input apply_se3_augmentation & color drop
        # SE(3) augmentation of point clouds and actions
        # NOTE: if use apply_se3_augmentation, then new action_trans, action_rot_grip(rot part), 
        # action_trans_continuous is from action_gripper_pose,
        # the old action_trans, action_rot_grip(rot part) is not be used except for its shape. 
        # action_rot_grip(grip part) is still from old action_rot_grip.
        # apply_se3_augmentation will also change quat
        # in apply_se3_augmentation, bound are only used for bound gripper pose(i.e. action), not used for bound point cloud
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd, action_trans_continuous, quat = apply_se3_augmentation(pcd,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device,
                                         )

        # color drop
        if self._color_drop > 0:
            assert obs[0][0].shape[0] == bs
            assert len(obs) == self._num_cameras
            for i in range(bs):
                if torch.rand(1) < self._color_drop:
                    for j in range(self._num_cameras):
                        obs[j][0][i, :] = 0

        # feature drop
        if self._feat_drop > 0:
            feat_mask = torch.ones(bs, device=self._device)
            for i in range(bs):
                if torch.rand(1) < self._feat_drop:
                    feat_mask[i] = 0
        else:
            feat_mask = None

        # (3) forward pass
        q_trans, q_rot_grip, q_collision, pred_dict = self._net(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            feat_mask,
            pcd_bound_masks=pcd_bound_masks)

        # (4) calculate loss
        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # (4.1) translation loss
        if self._trans_cls:
            # trans
            action_trans_x_one_hot = self._action_trans_x_one_hot_zeros.clone()
            action_trans_y_one_hot = self._action_trans_y_one_hot_zeros.clone()
            action_trans_z_one_hot = self._action_trans_z_one_hot_zeros.clone()

            for b in range(bs):
                gt_trans = action_trans[b, :].int()
                action_trans_x_one_hot[b, gt_trans[0]] = 1
                action_trans_y_one_hot[b, gt_trans[1]] = 1
                action_trans_z_one_hot[b, gt_trans[2]] = 1

            # flatten predictions
            q_trans_x_flat = q_trans[:, 0 * self._voxel_size : 1 * self._voxel_size]
            q_trans_y_flat = q_trans[:, 1 * self._voxel_size : 2 * self._voxel_size]
            q_trans_z_flat = q_trans[:, 2 * self._voxel_size : 3 * self._voxel_size]

            # trans loss
            q_trans_loss += self._celoss(q_trans_x_flat,
                                         action_trans_x_one_hot)
            q_trans_loss += self._celoss(q_trans_y_flat,
                                         action_trans_y_one_hot)
            q_trans_loss += self._celoss(q_trans_z_flat,
                                         action_trans_z_one_hot)

        else:
            if 'trans_per_point' in pred_dict.keys():
                # point prediction
                real_pos = pred_dict['real_pos']  # (B, N, 3)
                continuous_trans_pred_per_point = pred_dict['trans_per_point']  # (B, N, 3)
                q_trans_continuous = pred_dict['trans']
                N = real_pos.shape[1]
                continuous_trans_gt_per_point = action_trans_continuous.unsqueeze(
                    1).expand(-1, N, -1).clone()  # (B, N, 3)
                distance = torch.sqrt(
                    torch.sum((real_pos - continuous_trans_gt_per_point)**2,
                              dim=-1))  # (B, N)
                trans_loss_per_point_weight = F.softmax(
                    -distance / self._temperature_trans, dim=-1)  # (B, N)
                rot_grip_collision_loss_per_point_weight = F.softmax(
                    -distance / self._temperature_rot_grip_collision,
                    dim=-1)  # (B, N)
                if 'rot_grip_collision_out_per_point' in pred_dict:
                    rot_grip_collision_out_per_point = pred_dict[
                        'rot_grip_collision_out_per_point']
                    rot_grip_collision_out_logits = pred_dict[
                        'rot_grip_collision_out_logits']
                else:
                    rot_grip_collision_out_per_point = None
                    rot_grip_collision_out_logits = True
            else:
                # global prediction
                q_trans_continuous = pred_dict['trans']
                real_pos, continuous_trans_pred_per_point = None, None
                rot_grip_collision_out_per_point = None
                rot_grip_collision_out_logits = True

            q_trans_continuous_flat = q_trans_continuous.view(bs, -1)
            action_trans_continuous_flat = action_trans_continuous.view(bs, -1)
            if self._regression_loss.lower(
            ) == 'l2' or self._regression_loss.lower() == 'mse':
                # global trans loss
                q_trans_loss = self._mseloss(q_trans_continuous_flat,
                                             action_trans_continuous_flat)
                if real_pos is not None:
                    # point trans loss
                    trans_loss_per_point = torch.mean(
                        (continuous_trans_pred_per_point -
                         continuous_trans_gt_per_point)**2,
                        dim=-1)  # (B, N)
                    trans_loss_points = (torch.sum(trans_loss_per_point *
                                                   trans_loss_per_point_weight,
                                                   dim=1)).mean()
                    if not self._trans_point_loss:
                        trans_loss_points *= 0.0

            elif self._regression_loss.lower(
            ) == 'l1' or self._regression_loss.lower() == 'mae':
                # global trans loss
                q_trans_loss = self._l1loss(q_trans_continuous_flat,
                                            action_trans_continuous_flat)
                if real_pos is not None:
                    # point trans loss
                    trans_loss_per_point = torch.mean(
                        torch.abs(continuous_trans_pred_per_point -
                                  continuous_trans_gt_per_point),
                        dim=-1)  # (B, N)
                    trans_loss_points = (torch.sum(trans_loss_per_point *
                                                   trans_loss_per_point_weight,
                                                   dim=1)).mean()
                    if not self._trans_point_loss:
                        trans_loss_points *= 0.0

            else:
                raise NotImplementedError

        # (4.2) rot/grip/collision loss
        # rotation, gripper, and collision one-hots
        if self._rot_cls:
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
        action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
        action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone(
        )

        for b in range(bs):
            gt_rot_grip = action_rot_grip[b, :]
            if self._rot_cls:
                action_rot_x_one_hot[b, gt_rot_grip[0].int()] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1].int()] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2].int()] = 1

            action_grip_one_hot[b, gt_rot_grip[3].int()] = 1

            gt_ignore_collisions = action_ignore_collisions[b, :].int()
            action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

        # flatten predictions
        if self._rot_cls:
            q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 *
                                      self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 *
                                      self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 *
                                      self._num_rotation_classes]
        else:
            q_rot_quat = q_rot_grip[:, :4]

        q_grip_flat = q_rot_grip[:, -2:]
        q_ignore_collisions_flat = q_collision

        if rot_grip_collision_out_logits:
            # logit output
            # rotation loss
            if self._rot_cls:
                q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
                q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
                q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)
            else:
                q_rot_loss += self._quat_loss(q_rot_quat, quat)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat,
                                             action_ignore_collisions_one_hot)
        else:
            # softmax output
            # rotation loss
            q_rot_loss += self._celoss_softmax(q_rot_x_flat,
                                               action_rot_x_one_hot)
            q_rot_loss += self._celoss_softmax(q_rot_y_flat,
                                               action_rot_y_one_hot)
            q_rot_loss += self._celoss_softmax(q_rot_z_flat,
                                               action_rot_z_one_hot)
            # gripper loss
            q_grip_loss += self._celoss_softmax(q_grip_flat,
                                                action_grip_one_hot)
            # collision loss
            q_collision_loss += self._celoss_softmax(
                q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        if rot_grip_collision_out_per_point is not None:
            # rot_grip_collision point prediction
            rot_x_per_point = rot_grip_collision_out_per_point[
                :, :, self._num_rotation_classes * 0:self._num_rotation_classes * 1]
            rot_y_per_point = rot_grip_collision_out_per_point[
                :, :, self._num_rotation_classes * 1:self._num_rotation_classes * 2]
            rot_z_per_point = rot_grip_collision_out_per_point[
                :, :, self._num_rotation_classes * 2:self._num_rotation_classes * 3]
            grip_per_point = rot_grip_collision_out_per_point[
                :, :, self._num_rotation_classes * 3:self._num_rotation_classes * 3 + 2]
            collision_per_point = rot_grip_collision_out_per_point[:, :, -2:]

            action_rot_x_one_hot_per_point = action_rot_x_one_hot.unsqueeze(
                1).expand(-1, N, -1).clone()  # (B, N, num_rotation_classes)
            action_rot_y_one_hot_per_point = action_rot_y_one_hot.unsqueeze(
                1).expand(-1, N, -1).clone()
            action_rot_z_one_hot_per_point = action_rot_z_one_hot.unsqueeze(
                1).expand(-1, N, -1).clone()
            action_grip_one_hot_per_point = action_grip_one_hot.unsqueeze(
                1).expand(-1, N, -1).clone()
            action_ignore_collisions_one_hot_per_point = action_ignore_collisions_one_hot.unsqueeze(
                1).expand(-1, N, -1).clone()

            # rotation point loss
            rot_loss_per_point = self._pointwise_celoss(
                rot_x_per_point, action_rot_x_one_hot_per_point)  # (B, N)
            rot_loss_per_point += self._pointwise_celoss(
                rot_y_per_point, action_rot_y_one_hot_per_point)
            rot_loss_per_point += self._pointwise_celoss(
                rot_z_per_point, action_rot_z_one_hot_per_point)

            rot_diff_per_point = self._pointwise_diff(
                rot_x_per_point, action_rot_x_one_hot_per_point)
            rot_diff_per_point += self._pointwise_diff(
                rot_y_per_point, action_rot_y_one_hot_per_point)
            rot_diff_per_point += self._pointwise_diff(
                rot_z_per_point, action_rot_z_one_hot_per_point)
            rot_diff_per_point *= self._rotation_resolution

            # grip and collision point loss
            grip_loss_per_point = self._pointwise_celoss(
                grip_per_point, action_grip_one_hot_per_point)
            collision_loss_per_point = self._pointwise_celoss(
                collision_per_point,
                action_ignore_collisions_one_hot_per_point)

            # weighted sum
            rot_loss_points = (torch.sum(
                rot_loss_per_point * rot_grip_collision_loss_per_point_weight,
                dim=1)).mean()
            grip_loss_points = (torch.sum(
                grip_loss_per_point * rot_grip_collision_loss_per_point_weight,
                dim=1)).mean()
            collision_loss_points = (torch.sum(
                collision_loss_per_point *
                rot_grip_collision_loss_per_point_weight,
                dim=1)).mean()
            if not self._rot_point_loss:
                rot_loss_points *= 0.0
                grip_loss_points *= 0.0
                collision_loss_points *= 0.0

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                        (q_rot_loss * self._rot_loss_weight) + \
                        (q_grip_loss * self._grip_loss_weight) + \
                        (q_collision_loss * self._collision_loss_weight)

        if (not self._trans_cls) and real_pos is not None:
            combined_losses += trans_loss_points * self._trans_loss_weight

        if rot_grip_collision_out_per_point is not None:
            combined_losses += rot_loss_points * self._rot_loss_weight
            combined_losses += grip_loss_points * self._grip_loss_weight
            combined_losses += collision_loss_points * self._collision_loss_weight

        total_loss = combined_losses.mean()

        # (5) update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        # (6) summary, r_scheduler
        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean(),
            'losses/grip_loss': q_grip_loss.mean(),
            'losses/collision_loss': q_collision_loss.mean(),
        }

        if (not self._trans_cls) and real_pos is not None:
            self._summaries[
                'losses/trans_loss_points'] = trans_loss_points.mean()

        if rot_grip_collision_out_per_point is not None:
            self._summaries['losses/rot_loss_points'] = rot_loss_points.mean()
            self._summaries['losses/grip_loss_points'] = grip_loss_points.mean(
            )
            self._summaries[
                'losses/collision_loss_points'] = collision_loss_points.mean()
            self._summaries[
                'diff/rot_diff_per_point'] = rot_diff_per_point.mean()

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        return {
            'total_loss': total_loss,
        }

    def act(self,
            step: int,
            observation: dict,
            deterministic=False) -> ActResult:
        # What other agent do on act function:
        # preprocess_agent.py: normlize rgb to [-1,1]; (only use fisrt sample from buffer, i.e. t=1)
        # stack_agent.py: (on act) output continuous_action, discrete_euler_to_quaternion

        # (1) input
        bounds = self._coordinate_bounds
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(
                lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size

        # proprioception
        proprio = None
        if self._include_low_dim_state:
            proprio = observation['low_dim_state']

        obs, pcd = self._preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)]
               for o in obs]
        proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)

        if self._bound_pcd_before_transform:
            pcd_bound_masks = self.get_pcd_bound_masks(pcd, bounds)
        else:
            pcd_bound_masks = None

        feat_mask = None
        # (2) inference
        q_trans, q_rot_grip, q_ignore_collisions, pred_dict = self._net(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            feat_mask,
            pcd_bound_masks=pcd_bound_masks)

        # (3) get action
        q_trans_continuous = pred_dict['trans']
        if 'trans_per_point' in pred_dict.keys():
            if 'rot_grip_collision_out_per_point' in pred_dict:
                rot_grip_collision_out_logits = pred_dict[
                    'rot_grip_collision_out_logits']
            else:
                rot_grip_collision_out_logits = True
        else:
            rot_grip_collision_out_logits = True

        # softmax predictions
        q_trans = self._softmax_q_trans(
            q_trans) if q_trans is not None else q_trans

        if rot_grip_collision_out_logits:
            # q_rot_grip and q_ignore_collisions is logits
            # _softmax_q_rot_grip has been modified for self._rot_cls=0
            q_rot_grip = self._softmax_q_rot_grip(
                q_rot_grip) if q_rot_grip is not None else q_rot_grip
            q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
                if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        # choose_highest_action has been modified for self._rot_cls=0
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._net.choose_highest_action(
            q_trans, q_rot_grip, q_ignore_collisions, self._rot_cls)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int(
        ) if ignore_collisions is not None else None
        if self._trans_cls:
            coords = coords.int()
            # if translation is discrete, convert to continuous
            attention_coordinate = bounds[:, :3] + res * coords + res / 2
        else:
            # if translation is continuous, use the continuous translation
            attention_coordinate = q_trans_continuous

        observation_elements = {
            'attention_coordinate': attention_coordinate,
        }
        info = {
            'q_depth%d' % self._layer: q_trans,
            'voxel_idx_depth%d' % self._layer: coords
        }
        # NOTE: if self._trans_cls==False(continuous translation), then coords is None.
        # NOTE: The translation used for execution is extracted from 
        # attention_coordinate (continuous translation) instead of coords (discrete translation).
        return ActResult((coords, rot_grip_action, ignore_collisions_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        checkpoint = torch.load(weight_file, map_location=device)
        
        # Load model state dict
        merged_state_dict = self._net.state_dict()
        state_dict = checkpoint['model_state']
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_sgr_net.module', '_sgr_net')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                logging.warning("key %s not found in checkpoint" % k)
        self._net.load_state_dict(merged_state_dict)
        
        if self._training:
            # Load optimizer state if provided
            self._optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Load learning rate scheduler state if provided
            if self._lr_scheduler:
                self._scheduler.load_state_dict(checkpoint['lr_sched_state'])
        
        logging.info(f"Loaded weights from {weight_file}")


    def save_weights(self, savedir: str):
        save_dict = {
            'model_state': self._net.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'lr_sched_state': self._scheduler.state_dict() if self._lr_scheduler else None,
        }
        torch.save(save_dict, os.path.join(savedir, '%s.pt' % self._name))
