import copy
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List
from helpers.utils import PTH_PATH
from helpers.network_utils import DenseBlock, Conv2DBlock, Conv2DFiLMBlock, SiameseNet


class SemanticCNNLangAndFcsNet(nn.Module):

    def __init__(
        self,
        siamese_net_list: List[SiameseNet],
        num_cameras: int,
        use_semantic: List[int],
        pretrained_model: str,
        include_rgb: bool,
        semantic_dim_per_layer: int,
        num_rotation_classes: int,
        num_collision_classes: int,
        low_dim_state_len: int,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        clip_align: bool,
        align_type: str,
        network_input_image_size: int,
        norm: str = None,
        activation: str = 'relu',
        fc_layers: List[int] = None,
        no_language=False,
    ):
        super(SemanticCNNLangAndFcsNet, self).__init__()
        assert len(siamese_net_list) == num_cameras
        self._siamese_net_list = copy.deepcopy(siamese_net_list)
        self._num_cameras = num_cameras
        self._use_semantic = use_semantic
        self._pretrained_model = pretrained_model
        self._include_rgb = include_rgb
        self._semantic_dim_per_layer = semantic_dim_per_layer

        self._num_rotation_classes = num_rotation_classes
        self._num_collision_classes = num_collision_classes

        self._low_dim_state_len = low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides

        self._clip_align = clip_align
        self._align_type = align_type
        self._network_input_image_size = network_input_image_size

        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._no_language = no_language

        self._lang_feat_dim = 1024

    def build(self):
        for siamese_net in self._siamese_net_list:
            siamese_net.build()

        if len(self._use_semantic) > 0:
            if self._network_input_image_size > 128:
                self.input_upsample_transforms = T.Resize(
                    self._network_input_image_size,
                    interpolation=T.InterpolationMode.BILINEAR)

            if self._pretrained_model in ['moco', 'vicregl', 'simsiam', 'r3m']:
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

                elif self._pretrained_model == 'r3m':

                    def remove_language_head(state_dict):
                        keys = state_dict.keys()
                        ## Hardcodes to remove the language head
                        ## Assumes downstream use is as visual representation
                        for key in list(keys):
                            if ("lang_enc" in key) or ("lang_rew" in key):
                                del state_dict[key]
                            else:
                                state_dict[key[len("module.convnet."):]] = state_dict[key]
                                del state_dict[key]
                        return state_dict

                    state_dict = remove_language_head(checkpoint['r3m'])

                else:
                    raise NotImplementedError

                from helpers.resnet import resnet50
                # from torchvision.models import resnet50
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

            else:
                raise NotImplementedError

            if self._pretrained_model in ['moco', 'vicregl', 'simsiam', 'r3m']:
                self.norm_transforms = T.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])
            elif self._pretrained_model == 'clip':
                self.norm_transforms = T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
            self.resize_transforms = T.Resize(
                128, interpolation=T.InterpolationMode.BILINEAR)
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
            self.downsample_layer_list = nn.ModuleList()
            for i in self._use_semantic:
                self.downsample_layer_list.append(
                    Conv2DBlock(in_channels_list[i - 1],
                                self._semantic_dim_per_layer,
                                kernel_sizes=1,
                                strides=1,
                                norm=None,
                                activation='relu'))

        channels = self._siamese_net_list[0].output_channels
        self._camera_fuse = Conv2DBlock(channels * self._num_cameras, channels,
                                        1, 1, self._norm, self._activation)

        channels += self._low_dim_state_len
        self.conv1 = Conv2DFiLMBlock(channels, self._filters[0],
                                     self._kernel_sizes[0], self._strides[0])
        self.gamma1 = nn.Linear(self._lang_feat_dim, self._filters[0])
        self.beta1 = nn.Linear(self._lang_feat_dim, self._filters[0])

        self.conv2 = Conv2DFiLMBlock(self._filters[0], self._filters[1],
                                     self._kernel_sizes[1], self._strides[1])
        self.gamma2 = nn.Linear(self._lang_feat_dim, self._filters[1])
        self.beta2 = nn.Linear(self._lang_feat_dim, self._filters[1])

        self.conv3 = Conv2DFiLMBlock(self._filters[1], self._filters[2],
                                     self._kernel_sizes[2], self._strides[2])
        self.gamma3 = nn.Linear(self._lang_feat_dim, self._filters[2])
        self.beta3 = nn.Linear(self._lang_feat_dim, self._filters[2])

        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)
        self.tanh = nn.Tanh()

    def get_semantic_feature(self, rgb, lang_goal_emb, include_rgb=True):
        # rgb: [batch_size, 3, 128, 128]
        # preprocess, input range:[-1,1], dtype:torch.float32, device:cuda
        oringinal_rgb = rgb
        rgb = (rgb + 1) / 2  #[-1,1] -> [0,1]
        rgb = self.norm_transforms(rgb)
        if self._network_input_image_size > 128:
            rgb = self.input_upsample_transforms(rgb)

        with torch.no_grad():
            self.semantic_model.eval()
            if self._pretrained_model in ['moco', 'vicregl', 'simsiam', 'r3m']:
                _, layer_dict = self.semantic_model(rgb)
            elif self._pretrained_model == 'clip':
                _, im = self.semantic_model.prepool_im(rgb)
                if self._clip_align and (self._pretrained_model == 'clip'
                                         ) and (5 in self._use_semantic):
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
                    elif self._align_type == 'both':  # conbine vision*lang, sum(vision*lang)
                        im[-1] = torch.cat(
                            [im[-1],
                             torch.sum(im[-1], dim=1, keepdim=True)],
                            dim=1)
                    elif self._align_type == 'both2':  # conbine vison, sum(vision*lang)
                        im[-1] = torch.cat([
                            vision_im,
                            torch.sum(im[-1], dim=1, keepdim=True)
                        ], dim=1)
                    elif self._align_type == 'all':  # conbine vison, vision*lang, sum(vision*lang)
                        im[-1] = torch.cat([
                            vision_im, im[-1],
                            torch.sum(im[-1], dim=1, keepdim=True)
                        ], dim=1)

                im = [i.type(torch.cuda.FloatTensor) for i in im]
                layer_dict = {1: im[2], 2: im[4], 3: im[5], 4: im[6], 5: im[7]}

        embs = []
        for i, layer in enumerate(self._use_semantic):
            emb = self.downsample_layer_list[i](
                layer_dict[layer]
            )  # [batch_size, 1024, 32, 32] -> [batch_size, 64, 32, 32]
            emb = self.resize_transforms(emb)  # [batch_size, 64, 128, 128]
            embs.append(emb)
        embs = torch.cat(embs, dim=1)

        if include_rgb:
            embs = torch.cat([oringinal_rgb, embs],
                             dim=1)  #[batch_size, 3+64, 128, 128]
        return embs

    def pcd_normlize(self, pcd, repeat_bounds):
        min = repeat_bounds[:, :3, :, :]
        max = repeat_bounds[:, 3:, :, :]
        pcd = 2 * (pcd - (min + max) / 2) / (max - min)
        return pcd

    def forward(self, pcds, rgbs, bounds, low_dim_ins, lang_goal_emb):
        # pcds, rgbs: a list of [bs, 3, 128, 128]
        # bounds: [bs, 6]
        # low_dim_ins: [bs, 4]
        # lang_goal_emb: [bs, K]
        if self._no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)

        assert len(rgbs) == len(pcds) == self._num_cameras

        _, _, width, height = pcds[0].shape
        repeat_bounds = bounds.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, width, height)  # (bs,6, w, h)
        pcds = [self.pcd_normlize(p, repeat_bounds) for p in pcds]

        if len(self._use_semantic) > 0:
            rgbs = [
                self.get_semantic_feature(o,
                                          lang_goal_emb,
                                          include_rgb=self._include_rgb)
                for o in rgbs
            ]
        siamese_outputs = []
        for i in range(self._num_cameras):
            siamese_out = self._siamese_net_list[i]([rgbs[i], pcds[i]])
            siamese_outputs.append(siamese_out)
        siamese_outputs = torch.cat(siamese_outputs, dim=1)
        x = self._camera_fuse(siamese_outputs)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)

        g1 = self.gamma1(lang_goal_emb)
        b1 = self.beta1(lang_goal_emb)
        x = self.conv1(combined, g1, b1)

        g2 = self.gamma2(lang_goal_emb)
        b2 = self.beta2(lang_goal_emb)
        x = self.conv2(x, g2, b2)

        g3 = self.gamma3(lang_goal_emb)
        b3 = self.beta3(lang_goal_emb)
        x = self.conv3(x, g3, b3)

        x = self._maxp(x).squeeze(-1).squeeze(-1)
        x = self._fcs(x)

        continuous_trans_pred = self.tanh(x[:, :3])
        rot_and_grip_q = x[:, 3:-self._num_collision_classes]
        collision_q = x[:, -self._num_collision_classes:]

        min = bounds[:, :3]
        max = bounds[:, 3:]
        continuous_trans_pred = (max - min) / 2 * continuous_trans_pred + (
            max + min) / 2

        return None, rot_and_grip_q, collision_q, continuous_trans_pred
