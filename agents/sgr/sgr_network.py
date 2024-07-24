import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from helpers.utils import PTH_PATH
from helpers.network_utils import DenseBlock, Conv2DBlock
from helpers.point_utils import filter_and_sample_points
from openpoints.models import build_model_from_cfg


class SGRNetwork(nn.Module):

    def __init__(
        self,
        use_semantic,
        include_rgb,
        pretrained_model,
        semantic_downsample_norm,
        semantic_downsample_act,
        num_rotation_classes,
        num_grip_classes,
        num_collision_classes,
        num_points,
        rand_sample,
        preprocess,
        preprocess_norm,
        preprocess_activation,
        proprio_emb_dim,
        rgb_emb_dim,
        semantic_dim_per_layer,
        late_proprio_concat,
        resample,
        hierarchical,
        width,
        resnet_pos,
        clip_align,
        align_type,
        network_input_image_size,
        point_pretrain_apply,
        point_pretrain_frozen,
        point_pretrain_lang,
        cam_resolution,
        trans_cls,
        trans_num_classes_per_axis,  # only used when trans_cls=True
        trans_point_wise,
        point_relative_pos,
        trans_point_uniform_weight,
        rot_grip_collision_point_wise,
        shared_weight,
        rot_grip_collision_shared_weight,
        rot_cls,
        cfg_model,
    ):
        super().__init__()

        self._use_semantic = use_semantic
        self._include_rgb = include_rgb
        self._pretrained_model = pretrained_model
        self._num_rotation_classes = num_rotation_classes
        self._num_grip_classes = num_grip_classes
        self._num_collision_classes = num_collision_classes
        self._num_points = num_points
        self._rand_sample = rand_sample
        self._preprocess = preprocess
        self._late_proprio_concat = late_proprio_concat
        self._resample = resample
        self._hierarchical = hierarchical
        self._resnet_pos = resnet_pos
        self._clip_align = clip_align
        self._align_type = align_type
        self._network_input_image_size = network_input_image_size

        self._point_pretrain_apply = point_pretrain_apply
        self._point_pretrain_frozen = point_pretrain_frozen
        self._point_pretrain_lang = point_pretrain_lang

        self._cam_resolution = cam_resolution
        self._trans_cls = trans_cls
        self._trans_num_classes_per_axis = trans_num_classes_per_axis
        self._trans_point_wise = trans_point_wise
        self._point_relative_pos = point_relative_pos
        self._trans_point_uniform_weight = trans_point_uniform_weight
        self._rot_grip_collision_point_wise = rot_grip_collision_point_wise
        self._shared_weight = shared_weight
        self._rot_grip_collision_shared_weight = rot_grip_collision_shared_weight
        self._rot_cls = rot_cls

        self.point_network = build_model_from_cfg(cfg_model)

        if self._point_pretrain_apply:
            # use ULIP pretrained pointnext
            checkpoint = torch.load(PTH_PATH['ulip'])
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # only retain encoder part, remove cls part
                if k.startswith('module.point_encoder.encoder'):
                    # Remove prefix
                    state_dict[
                        k[len('module.point_encoder.'):]] = state_dict[k]
                # Delete renamed or unused k
                del state_dict[k]
            msg = self.point_network.load_state_dict(state_dict, strict=False)
            assert len(msg.unexpected_keys) == 0

        if len(self._use_semantic) > 0:
            if self._network_input_image_size != self._cam_resolution[0]:
                self.input_upsample_transforms = T.Resize(
                    self._network_input_image_size,
                    interpolation=T.InterpolationMode.BILINEAR)

            if self._pretrained_model in ['moco', 'vicregl', 'simsiam']:
                checkpoint = torch.load(PTH_PATH[self._pretrained_model])

                if self._pretrained_model == 'moco':
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        # Retain only encoder_q up to before the embedding layer
                        if k.startswith(
                                'module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
                            # Remove prefix
                            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                        # Delete renamed or unused k
                        del state_dict[k]

                elif self._pretrained_model == 'vicregl':
                    state_dict = checkpoint

                elif self._pretrained_model == 'simsiam':
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        # Retain only encoder_q up to before the embedding layer
                        if k.startswith('module.encoder') and not k.startswith(
                                'module.encoder.fc'):
                            # Remove prefix
                            state_dict[k[len("module.encoder."):]] = state_dict[k]
                        # Delete renamed or unused k
                        del state_dict[k]

                from helpers.resnet import resnet50
                #from torchvision.models import resnet50
                self.semantic_model = resnet50(pretrained=False,
                                               progress=False,
                                               stride=2)
                msg = self.semantic_model.load_state_dict(state_dict,
                                                          strict=False)
                del state_dict
                assert len(msg.missing_keys) == 0

            elif self._pretrained_model == 'clip':
                from helpers.clip.core.clip import build_model, load_clip
                model, _ = load_clip('RN50', jit=False)
                clip_model = build_model(model.state_dict())
                del model
                self.semantic_model = clip_model.visual

            elif self._pretrained_model == 'none':
                # densefusion
                from helpers.resnet import resnet50
                #from torchvision.models import resnet50
                self.semantic_model = resnet50(pretrained=False,
                                               progress=False,
                                               stride=2)
            else:
                raise NotImplementedError

            if self._pretrained_model in ['moco', 'vicregl', 'simsiam', 'none']:
                self.norm_transforms = T.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])
            elif self._pretrained_model == 'clip':
                self.norm_transforms = T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))

            self.resize_transforms = T.Resize(
                self._cam_resolution[0],
                interpolation=T.InterpolationMode.BILINEAR)

            in_channels_list = [64, 256, 512, 1024, 2048]
            if self._clip_align:
                if (self._pretrained_model
                        == 'clip') and (5 in self._use_semantic):
                    if self._align_type == 'mult':
                        in_channels_list[-1] = 1024
                    elif self._align_type == 'sum':
                        in_channels_list[-1] = 1
                    elif self._align_type == 'both' or self._align_type == 'both2':
                        in_channels_list[-1] = 1025
                    elif self._align_type == 'all':
                        in_channels_list[-1] = 1025 + 1024
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            self.out_channels_list = [
                width, width * 2, width * 4, width * 8, width * 16
            ]
            self.downsample_layer_list = nn.ModuleList()
            if self._hierarchical:
                for i in self._use_semantic:
                    self.downsample_layer_list.append(
                        Conv2DBlock(in_channels_list[i - 1],
                                    self.out_channels_list[i - self._resnet_pos],
                                    kernel_sizes=1,
                                    strides=1,
                                    norm=semantic_downsample_norm,
                                    activation=semantic_downsample_act))
            else:
                for i in self._use_semantic:
                    self.downsample_layer_list.append(
                        Conv2DBlock(in_channels_list[i - 1],
                                    semantic_dim_per_layer,
                                    kernel_sizes=1,
                                    strides=1,
                                    norm=semantic_downsample_norm,
                                    activation=semantic_downsample_act))
        # prepocess
        if not self._point_pretrain_apply:
            if self._preprocess:
                self.rgb_preprocess = DenseBlock(
                    3,
                    rgb_emb_dim,
                    norm=preprocess_norm,
                    activation=preprocess_activation)
                self.proprio_preprocess = DenseBlock(
                    4,
                    proprio_emb_dim,
                    norm=preprocess_norm,
                    activation=preprocess_activation)
                if self._late_proprio_concat:
                    self.proprio_preprocess2 = DenseBlock(
                        4,
                        proprio_emb_dim,
                        norm=preprocess_norm,
                        activation=preprocess_activation)
            else:
                raise NotImplementedError

        self.seg = ('decoder_args' in cfg_model)
        if self.seg:
            if (not self._trans_point_wise) or (
                    not self._rot_grip_collision_point_wise):
                self.global_maxp = nn.AdaptiveMaxPool1d(1)
                self.dense0 = DenseBlock(cfg_model['cls_args']['num_classes'],
                                         cfg_model['cls_args']['num_classes'],
                                         None,
                                         activation='relu')
                self.dense1 = DenseBlock(cfg_model['cls_args']['num_classes'],
                                         cfg_model['cls_args']['num_classes'],
                                         None,
                                         activation='relu')
                if self._trans_cls:
                    if (not self._trans_point_wise) and (
                            not self._rot_grip_collision_point_wise):
                        action_dim = (self._trans_num_classes_per_axis * 3 +
                                      self._num_rotation_classes * 3 +
                                      self._num_grip_classes +
                                      self._num_collision_classes)
                    else:
                        raise NotImplementedError
                else:
                    action_dim = 3 * (1 - (self._trans_point_wise > 0)) + (
                        self._num_rotation_classes * 3 +
                        self._num_grip_classes + self._num_collision_classes
                    ) * (1 - (self._rot_grip_collision_point_wise > 0))
                self.fc = DenseBlock(cfg_model['cls_args']['num_classes'],
                                     action_dim, None, None)

            if self._trans_point_wise:
                assert not self._trans_cls, 'Currently trans_point_wise does not support trans_cls'
                self.f_trans_stem = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    cfg_model['cls_args']['num_classes'],
                    None,
                    activation='relu')
                self.f_trans1 = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    cfg_model['cls_args']['num_classes'],
                    None,
                    activation='relu')
                self.f_trans2 = DenseBlock(
                    cfg_model['cls_args']['num_classes'], 3, None, None)
                self.f_attention1 = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    cfg_model['cls_args']['num_classes'],
                    None,
                    activation='relu')
                self.f_attention2 = DenseBlock(
                    cfg_model['cls_args']['num_classes'], 1, None, None)

            if self._rot_grip_collision_point_wise:
                assert self._trans_point_wise, 'Currently rot_grip_collision_point_wise require trans_point_wise'
                self.f_rot_grip_collision_stem = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    cfg_model['cls_args']['num_classes'],
                    None,
                    activation='relu')
                self.f_rot_grip_collision1 = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    cfg_model['cls_args']['num_classes'],
                    None,
                    activation='relu')
                self.f_rot_grip_collision2 = DenseBlock(
                    cfg_model['cls_args']['num_classes'],
                    self._num_rotation_classes * 3 + self._num_grip_classes +
                    self._num_collision_classes, None, None)
                if not self._shared_weight:
                    if self._rot_grip_collision_shared_weight:
                        self.f_rot_grip_collision_attention1 = DenseBlock(
                            cfg_model['cls_args']['num_classes'],
                            cfg_model['cls_args']['num_classes'],
                            None,
                            activation='relu')
                        self.f_rot_grip_collision_attention2 = DenseBlock(
                            cfg_model['cls_args']['num_classes'], 1, None,
                            None)
                    else:
                        self.f_rot_attention1 = DenseBlock(
                            cfg_model['cls_args']['num_classes'],
                            cfg_model['cls_args']['num_classes'],
                            None,
                            activation='relu')
                        self.f_rot_attention2 = DenseBlock(
                            cfg_model['cls_args']['num_classes'], 1, None,
                            None)
                        self.f_grip_attention1 = DenseBlock(
                            cfg_model['cls_args']['num_classes'],
                            cfg_model['cls_args']['num_classes'],
                            None,
                            activation='relu')
                        self.f_grip_attention2 = DenseBlock(
                            cfg_model['cls_args']['num_classes'], 1, None,
                            None)
                        self.f_collision_attention1 = DenseBlock(
                            cfg_model['cls_args']['num_classes'],
                            cfg_model['cls_args']['num_classes'],
                            None,
                            activation='relu')
                        self.f_collision_attention2 = DenseBlock(
                            cfg_model['cls_args']['num_classes'], 1, None,
                            None)

    def get_semantic_feature(self,
                             rgb,
                             lang_goal_emb,
                             include_rgb=True,
                             feat_mask=None):
        # rgb: [batch_size, 3, camera_resolution0, camera_resolution1]
        # preprocess, input range:[-1,1], dtype:torch.float32, device:cuda
        oringinal_rgb = rgb

        rgb = (rgb + 1) / 2  #[-1,1] -> [0,1]
        rgb = self.norm_transforms(rgb)
        if self._network_input_image_size != self._cam_resolution[0]:
            rgb = self.input_upsample_transforms(rgb)
        if self._pretrained_model in ['moco', 'vicregl', 'simsiam', 'clip']:
            with torch.no_grad():
                self.semantic_model.eval()
                if self._pretrained_model in ['moco', 'vicregl', 'simsiam']:
                    _, layer_dict = self.semantic_model(rgb)
                elif self._pretrained_model == 'clip':
                    _, im = self.semantic_model.prepool_im(rgb)
                    if self._clip_align and (5 in self._use_semantic):
                        im[-1] = im[-1].permute(0, 2, 3,
                                                1)  # [N,C,H,W] -> [N,H,W,C]
                        im[-1] = self.semantic_model.attnpool.v_proj(im[-1])
                        im[-1] = self.semantic_model.attnpool.c_proj(im[-1])
                        im[-1] = im[-1].permute(0, 3, 1,
                                                2)  # [N,H,W,C'] -> [N,C',H,W]
                        vision_im = copy.deepcopy(im[-1])
                        _, _, H, W = im[-1].shape
                        lang_goal_emb = lang_goal_emb.unsqueeze(2).unsqueeze(
                            3).repeat(1, 1, H, W)
                        im[-1] = lang_goal_emb * im[-1]
                        if self._align_type == 'sum':  # sum(vision*lang)
                            im[-1] = torch.sum(im[-1], dim=1, keepdim=True)
                        elif self._align_type == 'both':  # combine vision*lang, sum(vision*lang)
                            im[-1] = torch.cat([
                                im[-1],
                                torch.sum(im[-1], dim=1, keepdim=True)
                            ], dim=1)
                        elif self._align_type == 'both2':  # combine vison, sum(vision*lang)
                            im[-1] = torch.cat([
                                vision_im,
                                torch.sum(im[-1], dim=1, keepdim=True)
                            ], dim=1)
                        elif self._align_type == 'all':  # combine vison, vision*lang, sum(vision*lang)
                            im[-1] = torch.cat([
                                vision_im, im[-1],
                                torch.sum(im[-1], dim=1, keepdim=True)
                            ], dim=1)

                    im = [i.type(torch.cuda.FloatTensor) for i in im]
                    layer_dict = {
                        1: im[2],
                        2: im[4],
                        3: im[5],
                        4: im[6],
                        5: im[7]
                    }

        elif self._pretrained_model == 'none':
            # densefusion
            _, layer_dict = self.semantic_model(rgb)

        embs = []
        for i, layer in enumerate(self._use_semantic):
            emb = self.downsample_layer_list[i](layer_dict[layer])
            emb = self.resize_transforms(emb)
            embs.append(emb)
        embs = torch.cat(embs, dim=1)

        if feat_mask is not None:
            embs = embs * feat_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if include_rgb:
            embs = torch.cat(
                [oringinal_rgb, embs], dim=1
            )  # [batch_size, 3+C, camera_resolution0, camera_resolution1]
        return embs

    def forward(self,
                pcd,
                rgb,
                bounds,
                proprio,
                lang_goal_emb,
                lang_token_embs,
                feat_mask=None,
                pcd_bound_masks=None):
        bs = pcd[0].shape[0]

        if len(self._use_semantic) > 0:
            image_features = [
                self.get_semantic_feature(o,
                                          lang_goal_emb,
                                          include_rgb=self._include_rgb,
                                          feat_mask=feat_mask) for o in rgb
            ]
        elif not self._include_rgb:
            image_features = [
                torch.zeros(o.shape[0],
                            0,
                            o.shape[2],
                            o.shape[3],
                            device=o.device) for o in rgb
            ]
        else:
            image_features = rgb

        feat_size = image_features[0].shape[
            1]  # for rgb, feat_size=3, image_features[0].shape is [batch_size, 3, camera_resolution, camera_resolution]

        xyz, feature = filter_and_sample_points(image_features, pcd, feat_size,
                                                self._num_points, bs, bounds,
                                                self._rand_sample,
                                                self._resample,
                                                pcd_bound_masks)

        if not self._point_pretrain_apply:
            if self._hierarchical:
                # feature : [B, N, C]
                rgb_feature = feature[:, :, :3 * self._include_rgb]
                idx = 3 * self._include_rgb
                resnet_layer_dict = {}
                for layer in self._use_semantic:
                    # store the output of each layer of semantic feature into resnet_layer_dict
                    resnet_layer_dict[layer] = feature[:, :,idx: idx + self.out_channels_list[layer - self._resnet_pos]] \
                            .permute(0, 2, 1).contiguous()
                    idx += self.out_channels_list[layer - self._resnet_pos]
                assert idx == feature.shape[2]
                feature = rgb_feature

            if self._preprocess:
                if len(self._use_semantic) > 0:
                    if self._include_rgb:
                        rgb, semantic_feature = feature[:, :, :3], feature[:, :, 3:]
                        feature = torch.cat(
                            [self.rgb_preprocess(rgb), semantic_feature],
                            dim=-1)
                    else:
                        feature = feature
                else:
                    if self._include_rgb:
                        feature = self.rgb_preprocess(feature)  # [B, N, C]

        xyz = xyz.permute(0, 2, 1)  # [B, 3, N]
        feature = feature.permute(0, 2, 1)  # [B, C, N]

        N = xyz.shape[2]
        if not self._point_pretrain_apply:
            if self._preprocess:
                proprio1 = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            else:
                proprio1 = proprio
            proprio1 = proprio1.unsqueeze(-1).repeat(1, 1, N)
            feature = torch.cat([feature, proprio1], dim=1)  # [B, C, N]

        # normlize
        repeat_bounds = bounds.unsqueeze(-1).repeat(1, 1, N)  # [B, 6, N]
        min = repeat_bounds[:, :3, :]
        max = repeat_bounds[:, 3:, :]
        xyz = 2 * (xyz - (min + max) / 2) / (max - min)

        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        if self._point_pretrain_apply:
            # ULIP pretrained pointnext
            feature = xyz.clone().permute(0, 2, 1)  # [B, 3, N]
            height = torch.zeros([bs, 1, N]).to(feature.device)
            height_dim = 2  # i.e. Z axis is height
            for b in range(bs):
                height[b, 0, :] = feature[b, height_dim, :] - torch.min(
                    feature[b, height_dim, :])

            feature = torch.cat(
                [feature, height], dim=1
            )  # [B, 4, N], ULIP use xyz + height as feature, not use RGB

            data = {'pos': xyz.contiguous(), 'x': feature.contiguous()}
            if self._point_pretrain_lang:
                prediction_concat_content = torch.cat([lang_goal_emb, proprio],
                                                      dim=-1)
            else:
                prediction_concat_content = proprio
            x = self.point_network(data,
                                   prediction_concat_content,
                                   frozen_encoder=self._point_pretrain_frozen)
        else:
            data = {
                'pos': xyz.contiguous(),
                'x': feature.contiguous(),
                'lang_goal_emb': lang_goal_emb
            }
            if self._hierarchical:
                data['resnet_layer_dict'] = resnet_layer_dict

            if self._late_proprio_concat:
                proprio2 = self.proprio_preprocess2(proprio)
                x = self.point_network(data, proprio2)
            else:
                x = self.point_network(data)

        if self.seg:
            # input x: (B, C, N)
            if self._trans_point_wise:
                f_point_trans = x.permute(0, 2, 1)  # shape: (B, N, C)
                f_point_trans = self.f_trans_stem(f_point_trans)
                translation_offset = self.f_trans2(
                    self.f_trans1(f_point_trans))  # (B, N, 3)
                pos = data['pos'].clone().contiguous()  # shape: (B, N, 3)
                assert pos.shape == translation_offset.shape
                if self._point_relative_pos:
                    translation_per_point = translation_offset + pos  # (B, N, 3)
                else:
                    translation_per_point = translation_offset

                translation_per_point = translation_per_point.clamp(min=-1,
                                                                    max=1)
                if self._trans_point_uniform_weight:
                    trans_attention_weights = torch.full(
                        (bs, N, 1),
                        fill_value=1 / N,
                        device=translation_per_point.device)
                else:
                    trans_attention_weights = F.softmax(self.f_attention2(
                        self.f_attention1(f_point_trans)), dim=1)  # (B, N, 1)

                trans = torch.sum(trans_attention_weights * translation_per_point, dim=1) # (B, 3)

            if self._rot_grip_collision_point_wise:
                f_point_rot_grip_collision = x.permute(0, 2, 1)  # shape: (B, N, C)
                f_point_rot_grip_collision = self.f_rot_grip_collision_stem(
                    f_point_rot_grip_collision)
                rot_grip_collision_out_per_point = self.f_rot_grip_collision2(
                    self.f_rot_grip_collision1(f_point_rot_grip_collision))
                assert rot_grip_collision_out_per_point.shape[2] == (
                    self._num_rotation_classes * 3 + self._num_grip_classes +
                    self._num_collision_classes)

                rot_x_per_point = F.softmax(
                    rot_grip_collision_out_per_point[
                        :, :, self._num_rotation_classes * 0:self._num_rotation_classes * 1],
                    dim=-1)
                rot_y_per_point = F.softmax(
                    rot_grip_collision_out_per_point[
                        :, :, self._num_rotation_classes * 1:self._num_rotation_classes * 2],
                    dim=-1)
                rot_z_per_point = F.softmax(
                    rot_grip_collision_out_per_point[
                        :, :, self._num_rotation_classes * 2:self._num_rotation_classes * 3],
                    dim=-1)
                grip_per_point = F.softmax(
                    rot_grip_collision_out_per_point[
                        :, :, self._num_rotation_classes * 3:self._num_rotation_classes * 3 + self._num_grip_classes],
                    dim=-1)
                collision_per_point = F.softmax(
                    rot_grip_collision_out_per_point[
                        :, :, -self._num_collision_classes:],
                    dim=-1)
                rot_grip_collision_out_per_point_softmax = torch.cat(
                    [
                        rot_x_per_point, rot_y_per_point, rot_z_per_point,
                        grip_per_point, collision_per_point
                    ],
                    dim=-1)  # (B, N, C)

                if self._shared_weight:
                    rot_grip_collision_attention_weights = trans_attention_weights
                else:
                    if self._rot_grip_collision_shared_weight:
                        rot_grip_collision_attention_weights = F.softmax(
                            self.f_rot_grip_collision_attention2(
                                self.f_rot_grip_collision_attention1(
                                    f_point_rot_grip_collision)),
                            dim=1)  # (B, N, 1)
                    else:
                        rot_attention_weights = F.softmax(
                            self.f_rot_attention2(
                                self.f_rot_attention1(
                                    f_point_rot_grip_collision)),
                            dim=1)  # (B, N, 1)
                        grip_attention_weights = F.softmax(
                            self.f_grip_attention2(
                                self.f_grip_attention1(
                                    f_point_rot_grip_collision)),
                            dim=1)  # (B, N, 1)
                        collision_attention_weights = F.softmax(
                            self.f_collision_attention2(
                                self.f_collision_attention1(
                                    f_point_rot_grip_collision)),
                            dim=1)  # (B, N, 1)

                if self._shared_weight or self._rot_grip_collision_shared_weight:
                    rot_grip_collision_out_weighted = torch.sum(
                        rot_grip_collision_attention_weights *
                        rot_grip_collision_out_per_point_softmax,
                        dim=1)  # (B, C)
                else:
                    rot_out_per_point_softmax = rot_grip_collision_out_per_point_softmax[
                        :, :, self._num_rotation_classes * 0: self._num_rotation_classes * 3]
                    rot_out_weighted = torch.sum(
                        rot_attention_weights * rot_out_per_point_softmax,
                        dim=1)  # (B, 3*um_rotation_classes)
                    grip_out_per_point_softmax = rot_grip_collision_out_per_point_softmax[
                        :, :, self._num_rotation_classes * 3: self._num_rotation_classes * 3 + self._num_grip_classes]
                    grip_out_weighted = torch.sum(grip_attention_weights *
                                                  grip_out_per_point_softmax,
                                                  dim=1)  # (B, 2)
                    collision_out_per_point_softmax = rot_grip_collision_out_per_point_softmax[
                        :, :, -self._num_collision_classes:]
                    collision_out_weighted = torch.sum(
                        collision_attention_weights *
                        collision_out_per_point_softmax,
                        dim=1)  # (B, 2)
                    rot_grip_collision_out_weighted = torch.cat(
                        [
                            rot_out_weighted, grip_out_weighted,
                            collision_out_weighted
                        ],
                        dim=-1)  # (B, C)

            if (not self._trans_point_wise) or (
                    not self._rot_grip_collision_point_wise):
                x = self.global_maxp(x).view(bs, -1)  # (B, C)
                x = self.dense0(x)
                x = self.dense1(x)
                x = self.fc(x)
                if self._trans_point_wise:
                    x = torch.cat([trans, x], dim=-1)
            else:
                x = torch.cat([trans, rot_grip_collision_out_weighted], dim=-1)

        if self._trans_cls:
            assert self._rot_cls, 'not implemented'
            q_trans = x[:, :3 * self._trans_num_classes_per_axis]
            rot_and_grip_q = x[:, 3 * self._trans_num_classes_per_axis:-self._num_collision_classes]
            collision_q = x[:, -self._num_collision_classes:]
            pred_dict = {'trans': None}
        else:
            if self._rot_cls:
                # rot classification, rot is represneted by discrete euler angles
                assert x.shape[
                    1] == 3 + self._num_rotation_classes * 3 + self._num_grip_classes + self._num_collision_classes
            else:
                # rot regression, rot is represneted by quaternion
                assert x.shape[
                    1] == 3 + 4 + self._num_grip_classes + self._num_collision_classes

            trans = (x[:, :3]).clamp(min=-1, max=1)

            rot_and_grip_q = x[:, 3:-self._num_collision_classes]  # if rot_cls, num_rotation_classes * 3 + 2, else 4 + 2
            collision_q = x[:, -self._num_collision_classes:]

            min = bounds[:, :3]
            max = bounds[:, 3:]
            trans = (max - min) / 2 * trans + (max + min) / 2  # unnormlize
            pred_dict = {'trans': trans}
            if self.seg and self._trans_point_wise:
                min = min.unsqueeze(1).expand(-1, N, -1)
                max = max.unsqueeze(1).expand(-1, N, -1)
                real_pos = (max - min) / 2 * pos + (max + min) / 2
                continuous_trans_pred_per_point = (
                    max - min) / 2 * translation_per_point + (max + min) / 2

                pred_dict['real_pos'] = real_pos
                pred_dict['trans_per_point'] = continuous_trans_pred_per_point

                pred_dict['rot_grip_collision_out_per_point'] = rot_grip_collision_out_per_point
                pred_dict['rot_grip_collision_out_logits'] = False

            q_trans = None

        return q_trans, rot_and_grip_q, collision_q, pred_dict
