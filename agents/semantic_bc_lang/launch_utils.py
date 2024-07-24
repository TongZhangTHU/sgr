import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List
from omegaconf import DictConfig, ListConfig
from torch.multiprocessing import Process, Value, Manager

import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.network_utils import SiameseNet
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import build_model, load_clip, tokenize
from agents.semantic_bc_lang.networks import SemanticCNNLangAndFcsNet
from agents.semantic_bc_lang.semantic_bc_lang_agent import SemanticBCLangAgent
from agents.semantic_bc_lang.stack_agent import StackAgent

REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


def create_replay(batch_size: int,
                  timesteps: int,
                  save_dir: str,
                  cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    #language
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE, ), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32)
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 
    # 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size, ),
                      np.int32),
        ReplayElement('trans_action_continuous', (trans_indicies_size, ),
                      np.float32),
        ReplayElement('rot_grip_action_indicies',
                      (rot_and_grip_indicies_size, ), np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size, ),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size, ), np.float32),
        #language
        ReplayElement('lang_goal_emb', (lang_feat_dim, ), np.float32),
        ReplayElement('lang_token_embs', (
            max_token_seq_len,
            lang_emb_dim,
        ), np.float32),  # extracted from CLIP's language encoder
        ReplayElement('task', (), str),
        ReplayElement(
            'lang_goal', (1, ),
            object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8, ),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements)
    return replay_buffer


def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
        voxel_sizes: List[int],
        rotation_resolution: int):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = utils.correct_rotation_instability(disc_rot,
                                                  rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    trans_continuous = []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes):  # only single voxelization-level is used
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size,
                                           bounds)
        trans_indicies.extend(index.tolist())
        trans_continuous.extend(obs_tp1.gripper_pose[:3].tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose,
         np.array([grip])]), attention_coordinates, trans_continuous


def _add_keypoints_to_replay(cfg: DictConfig,
                             task: str,
                             replay: ReplayBuffer,
                             inital_obs: Observation,
                             demo: Demo,
                             episode_keypoints: List[int],
                             cameras: List[str],
                             rlbench_scene_bounds: List[float],
                             voxel_sizes: List[int],
                             rotation_resolution: int,
                             description: str = '',
                             clip_model=None,
                             device='cpu'):
    prev_action = None
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates, trans_continuous = \
            _get_action(obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, rotation_resolution)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = utils.extract_obs(obs,
                                     t=k,
                                     prev_action=prev_action,
                                     cameras=cameras,
                                     episode_length=cfg.rlbench.episode_length)
        #language
        tokens = tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'trans_action_continuous': trans_continuous,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'task': task,
            #language
            'lang_goal': np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1,
                                     t=k + 1,
                                     prev_action=prev_action,
                                     cameras=cameras,
                                     episode_length=cfg.rlbench.episode_length)
    #language
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],
                voxel_sizes: List[int],
                rotation_resolution: int,
                clip_model=None,
                device='cpu',
                keypoint_method='heuristic',
                variation_number=-1):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)

    variation_numbers = variation_number if isinstance(
        variation_number, ListConfig) or isinstance(
            variation_number, list) else [variation_number]
    for variation_number in variation_numbers:
        for d_idx in range(num_demos):
            # load demo from disk
            demo = rlbench_utils.get_stored_demos(
                amount=1,
                image_paths=False,
                dataset_root=cfg.rlbench.demo_path,
                variation_number=variation_number,
                task_name=task,
                obs_config=obs_config,
                random_selection=False,
                from_episode_number=d_idx)[0]

            descs = demo._observations[0].misc['descriptions']

            # extract keypoints (a.k.a keyframes)
            episode_keypoints = demo_loading_utils.keypoint_discovery(
                demo, method=keypoint_method)

            if rank == 0:
                logging.info(
                    f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task} - {variation_number}"
                )

            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                    episode_keypoints = episode_keypoints[1:]
                if len(episode_keypoints) == 0:
                    break
                _add_keypoints_to_replay(cfg,
                                         task,
                                         replay,
                                         obs,
                                         demo,
                                         episode_keypoints,
                                         cameras,
                                         rlbench_scene_bounds,
                                         voxel_sizes,
                                         rotation_resolution,
                                         description=desc,
                                         clip_model=clip_model,
                                         device=device)
        assert d_idx == num_demos - 1  # make sure that all data are loaded correctly
        logging.debug('Replay %s variation % s filled with demos.' %
                      (task, variation_number))


def fill_multi_task_replay(
    cfg: DictConfig,
    obs_config: ObservationConfig,
    rank: int,
    replay: ReplayBuffer,
    tasks: List[str],
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    clip_model=None,
    keypoint_method='heuristic',
    variation_number=-1,
):
    manager = Manager()
    store = manager.dict()

    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(
        store)  # if use_disk=False, here will run too much time, may bug
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            # model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
            #                             if torch.cuda.is_available() else 'cpu')
            model_device = torch.device('cuda:%s' %
                                        (rank % torch.cuda.device_count()) if
                                        torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay,
                        args=(cfg, obs_config, rank, replay, task, num_demos,
                              demo_augmentation, demo_augmentation_every_n,
                              cameras, rlbench_scene_bounds, voxel_sizes,
                              rotation_resolution, clip_model, model_device,
                              keypoint_method, variation_number))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def create_agent(cfg: DictConfig):
    num_cameras = len(cfg.rlbench.cameras)
    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    if cfg.method.include_rgb and cfg.method.rgb_emb_dim != 3:
        raise NotImplementedError
    feature_channel = (
        cfg.method.rgb_emb_dim * cfg.method.include_rgb + 
        cfg.method.semantic_dim_per_layer * len(cfg.method.use_semantic)
    )

    siamese_net_list = nn.ModuleList()
    for i in range(num_cameras):
        siamese_net = SiameseNet(
            input_channels=[feature_channel, 3],  # semantic_dim: 64, xyz_dim:3
            filters=[64],  #[16]
            kernel_sizes=[5],
            strides=[1],
            activation='relu',
            norm=None,
        )
        siamese_net_list.append(siamese_net)

    encoder = SemanticCNNLangAndFcsNet(
        siamese_net_list=siamese_net_list,
        num_cameras=num_cameras,
        use_semantic=cfg.method.use_semantic,
        pretrained_model=(cfg.method.pretrained_model).lower(),
        include_rgb=cfg.method.include_rgb,
        semantic_dim_per_layer=cfg.method.semantic_dim_per_layer,
        num_rotation_classes=num_rotation_classes,
        num_collision_classes=2,
        low_dim_state_len=LOW_DIM_SIZE,
        filters=[128, 256, 512],  #[32, 64, 64]
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        clip_align=cfg.method.clip_align,
        align_type=cfg.method.align_type,
        network_input_image_size=cfg.method.network_input_image_size,
        norm=None,
        activation='relu',
        fc_layers=[512, 256, 3 + 72 * 3 + 2 + 2],
        no_language=cfg.method.no_language
        )

    bc_agent = SemanticBCLangAgent(
        coordinate_bounds=cfg.rlbench.scene_bounds,
        encoder=encoder,
        camera_names=cfg.rlbench.cameras,
        batch_size=cfg.replay.batch_size,
        voxel_size=cfg.method.voxel_sizes[0],
        num_rotation_classes=num_rotation_classes,
        rotation_resolution=cfg.method.rotation_resolution,
        lr=cfg.method.lr,
        trans_loss_weight=cfg.method.trans_loss_weight,
        rot_loss_weight=cfg.method.rot_loss_weight,
        grip_loss_weight=cfg.method.grip_loss_weight,
        collision_loss_weight=cfg.method.collision_loss_weight,
        lambda_weight_l2=cfg.method.lambda_weight_l2,
        transform_augmentation=cfg.method.transform_augmentation.apply_se3,
        transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
        transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
        transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
        optimizer_type=cfg.method.optimizer,
        trans_cls=cfg.method.trans_cls,
        regression_loss=cfg.method.regression_loss,
        grad_clip=cfg.method.grad_clip,
        )
    bc_agents = StackAgent(
        qattention_agents=[bc_agent],
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    return PreprocessAgent(pose_agent=bc_agents)

