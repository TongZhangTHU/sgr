import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
from yarr.agents.agent import Agent, ActResult, ScalarSummary, Summary

from voxel.augmentation import apply_se3_augmentation
from helpers.clip.core.clip import build_model, load_clip
from helpers.optim.lamb import Lamb

NAME = 'SemanticBCLangAgent'


class QFunction(nn.Module):

    def __init__(self, encoder: nn.Module, rotation_resolution: float, device,
                 training):
        super(QFunction, self).__init__()
        self._qnet = encoder
        self._rotation_resolution = rotation_resolution
        self._qnet.build()
        self._qnet.to(device)

        # distributed training
        if training:
            # sync_bn
            self._qnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self._qnet)
            if encoder._use_semantic:
                self._qnet = DDP(self._qnet,
                                 device_ids=[device],
                                 find_unused_parameters=True)
            else:
                self._qnet = DDP(self._qnet,
                                 device_ids=[device],
                                 find_unused_parameters=False)

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans) if q_trans is not None else None
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(q_rot_grip[:, :-2],
                                            int(360 // self._rotation_resolution),
                                            dim=1),
                                dim=1)
            rot_and_grip_indicies = torch.cat([
                q_rot[:, 0:1].argmax(-1), q_rot[:, 1:2].argmax(-1),
                q_rot[:, 2:3].argmax(-1), q_rot_grip[:, -2:].argmax(-1, keepdim=True)
            ], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                rgb_pcd,
                proprio,
                pcd,
                lang_goal_emb,
                lang_token_embs,
                bounds=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]

        # pcd in rgb_pcd cannot be used, it hasn't been augmentation
        rgb = [rp[0] for rp in rgb_pcd]

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        q_trans, q_rot_and_grip, q_ignore_collisions, q_trans_continuous = self._qnet(
            pcd,
            rgb,
            bounds,
            proprio,
            lang_goal_emb,
            #lang_token_embs,
        )

        return q_trans, q_rot_and_grip, q_ignore_collisions, q_trans_continuous


class SemanticBCLangAgent(Agent):

    def __init__(
        self,
        coordinate_bounds: list,
        encoder: nn.Module,
        camera_names: list,
        batch_size: int,
        voxel_size: int,
        num_rotation_classes: int,
        rotation_resolution: float,
        lr: float = 0.0001,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
        transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = 'adam',
        trans_cls: bool = False,
        regression_loss: str = 'l2',
        grad_clip: float = None,
    ):
        self._coordinate_bounds_list = coordinate_bounds
        self._encoder = encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._lr = lr
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
        self._optimizer_type = optimizer_type

        self._trans_cls = trans_cls
        self._regression_loss = regression_loss
        self._grad_clip = grad_clip

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._mse_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._name = NAME

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        self._q = QFunction(self._encoder, self._rotation_resolution, device,
                            training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds_list,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            elif self._optimizer_type == 'adamw':
                self._optimizer = torch.optim.AdamW(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # one-hot zero tensors
            if self._trans_cls:
                self._action_trans_one_hot_zeros = torch.zeros(
                    (self._batch_size, 1, self._voxel_size, self._voxel_size,
                     self._voxel_size),
                    dtype=int,
                    device=device)
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
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._q.to(device)

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _mseloss(self, pred, labels):
        return self._mse_loss(pred, labels)

    def _l1loss(self, pred, labels):
        return self._l1_loss(pred, labels)

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 *
                                  self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 *
                                  self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 *
                                  self._num_rotation_classes]
        q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([
            q_rot_x_flat_softmax, q_rot_y_flat_softmax, q_rot_z_flat_softmax,
            q_grip_flat_softmax
        ], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def update(self, step: int, replay_sample: dict) -> dict:
        action_trans = replay_sample['trans_action_indicies'].int()
        action_trans_continuous = replay_sample[
            'trans_action_continuous'].float(
            )  # if self._transform_augmentation, this will not be used
        action_rot_grip = replay_sample['rot_grip_action_indicies'].int()
        action_gripper_pose = replay_sample['gripper_pose']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        #language
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()

        device = self._device
        bounds = self._coordinate_bounds.to(device)
        proprio = replay_sample['low_dim_state']
        obs, pcd = self._preprocess_inputs(replay_sample)
        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        # NOTE: here action_trans_continuous is from apply_se3_augmentation, not from replay_sample
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd, action_trans_continuous, _ = apply_se3_augmentation(pcd,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         0,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        # forward pass
        q_trans, q_rot_grip, q_collision, q_trans_continuous = self._q(
            obs, proprio, pcd, lang_goal_emb, lang_token_embs, bounds)

        # argmax to choose best action
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        q_trans_continuous_flat = q_trans_continuous.view(bs, -1)
        action_trans_continuous_flat = action_trans_continuous.view(bs, -1)
        if self._regression_loss.lower(
        ) == 'l2' or self._regression_loss.lower() == 'mse':
            q_trans_loss = self._mseloss(q_trans_continuous_flat,
                                         action_trans_continuous_flat)
        elif self._regression_loss.lower(
        ) == 'l1' or self._regression_loss.lower() == 'mae':
            q_trans_loss = self._l1loss(q_trans_continuous_flat,
                                        action_trans_continuous_flat)
        else:
            raise NotImplementedError

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone(
            )

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b,
                                                 gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 *
                                      self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 *
                                      self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 *
                                      self._num_rotation_classes]
            q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat,
                                             action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        if self._grad_clip is not None and self._q.parameters() is not None:
            nn.utils.clip_grad_value_(self._q.parameters(), self._grad_clip)
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
        }

        return {
            'total_loss': total_loss,
        }

    def act(self,
            step: int,
            observation: dict,
            deterministic=False) -> ActResult:

        bounds = self._coordinate_bounds
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        # language
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(
                lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size

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

        # inference
        q_trans, q_rot_grip, q_ignore_collisions, q_trans_continuous = self._q(
            obs, proprio, pcd, lang_goal_emb, lang_token_embs, bounds)

        # softmax Q predictions
        q_trans = self._softmax_q_trans(
            q_trans) if q_trans is not None else q_trans
        q_rot_grip = self._softmax_q_rot_grip(
            q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
            if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int(
        ) if ignore_collisions is not None else None
        if self._trans_cls:
            coords = coords.int()
            attention_coordinate = bounds[:, :3] + res * coords + res / 2
        else:
            attention_coordinate = q_trans_continuous

        observation_elements = {
            'attention_coordinate': attention_coordinate,
        }
        info = {'q': q_trans, 'voxel_idx': coords}
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
        device = self._device if not self._training else torch.device(
            'cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        logging.info("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(self._q.state_dict(),
                   os.path.join(savedir, '%s.pt' % self._name))
