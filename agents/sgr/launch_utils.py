import os
import torch
import logging
import numpy as np
from typing import List
from torch.multiprocessing import Process, Value, Manager
from omegaconf import DictConfig, OmegaConf, ListConfig

from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.utils import setup_logger
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize, build_model, load_clip, tokenize
from agents.sgr.sgr_network import SGRNetwork
from agents.sgr.sgr_agent import SGRAgent
from agents.sgr.stack_agent import StackAgent
from openpoints.utils import EasyConfig

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
            ObservationElement('%s_rgb' % cname, (
                3,
                *image_size,
            ), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32)
        )  
        # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (
                4,
                4,
            ), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (
                3,
                3,
            ), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision,
    # 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size, ),
                      np.int32),
        ReplayElement('trans_action_continuous', (trans_indicies_size, ),
                      np.float32),
        ReplayElement('rot_grip_action_indicies',
                      (rot_and_grip_indicies_size, ), np.int32),
        ReplayElement('quat', (4, ), np.float32),
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
    rotation_resolution: int,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    # disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    trans_continuous = []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    # for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used
    vox_size = voxel_sizes[0]
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
         np.array([grip])]), attention_coordinates, trans_continuous, quat


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
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates, trans_continuous, quat = \
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
            'quat': quat,
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
    # ignore_collisions is extracted from the below step
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
    if rank == 0:
        setup_logger(file_name='train.log',
                     logging_level=cfg.framework.logging_level)

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
            if cfg.real_robot is None:
                simulation = True
            else:
                simulation = not cfg.real_robot.apply
            if simulation:
                # simulation
                demo = rlbench_utils.get_stored_demos(
                    amount=1,
                    image_paths=False,
                    dataset_root=cfg.rlbench.demo_path,
                    variation_number=variation_number,
                    task_name=task,
                    obs_config=obs_config,
                    random_selection=False,
                    from_episode_number=d_idx)[0]
            else:
                # real robot
                demo = rlbench_utils.get_stored_demos(
                    amount=1,
                    image_paths=False,
                    dataset_root=cfg.rlbench.demo_path,
                    variation_number=variation_number,
                    task_name=task,
                    obs_config=obs_config,
                    random_selection=False,
                    from_episode_number=d_idx,
                    simulation=simulation,
                    real_robot_cfg=cfg.real_robot,
                )[0]
            descs = demo._observations[0].misc['descriptions']

            # extract keypoints (a.k.a keyframes)
            if simulation:
                episode_keypoints = demo_loading_utils.keypoint_discovery(
                    demo, method=keypoint_method)
            else:
                episode_keypoints = demo._observations[0].misc['keypoint_idxs']

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
        store)  # if use_disk=False, here will run too much time, maybe a bug
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
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


def str_none(str):
    return None if str.lower() == 'none' else str


def create_agent(cfg: DictConfig):

    cam_resolution = cfg.rlbench.camera_resolution
    if cam_resolution[0] != cam_resolution[1]:
        raise NotImplementedError("Only square images are supported")
    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    agents = []
    if not cfg.method.use_semantic:
        cfg.method.use_semantic = []  # false -> []

    if cfg.method.point_pretrain.apply:
        assert cfg.method.num_points == 8192  # ULIP is pretrained on 8192 points
        cfg_m = EasyConfig()
        cfg_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            cfg.method.point_pretrain.conf_path)
        cfg_m.load(cfg_path, recursive=True)
        logging.info(f"Point pretrain must use given cfg on : {cfg_path}")
        if cfg.method.point_pretrain.lang:
            cfg_m.model.cls_args.prediction_concat_dim = cfg.method.point_pretrain.lang_dim + LOW_DIM_SIZE
        else:
            cfg_m.model.cls_args.prediction_concat_dim = LOW_DIM_SIZE
        cfg_model = cfg_m.model

    else:
        rgb_emb_dim = cfg.method.rgb_emb_dim if cfg.method.preprocess else 3
        feature_channel = rgb_emb_dim * cfg.method.include_rgb + cfg.method.proprio_emb_dim + (
            cfg.method.semantic_dim_per_layer *
            len(cfg.method.use_semantic)) * (1 - cfg.method.hierarchical)

        cfg_model = EasyConfig()
        cfg_model.update(OmegaConf.to_container(cfg.model))
        cfg_model['encoder_args']['in_channels'] = feature_channel
        if cfg.method.late_proprio_concat:
            cfg_model['cls_args'][
                'prediction_concat_dim'] = cfg.method.proprio_emb_dim
        cfg_model['encoder_args']['resnet_layer_index'] = cfg.method.use_semantic if cfg.method.hierarchical else []

    if 'decoder_args' in cfg_model:
        logging.info('Network architecture: Encoder-Decoder.')
    else:
        logging.info('Network architecture: Encoder-Only.')
        if cfg.method.trans_cls:
            # trans classification
            cfg_model['cls_args']['num_classes'] = int(
                3 * cfg.method.voxel_sizes[0] + 3 * num_rotation_classes + 2 +
                2)
        else:
            # trans regression
            cfg_model['cls_args']['num_classes'] = int(3 + 3 *
                                                       num_rotation_classes +
                                                       2 + 2)

    if 'encoder_args' in cfg_model:
        if 'resnet_pos' in cfg_model['encoder_args']:
            resnet_pos = cfg_model['encoder_args']['resnet_pos']
        else:
            resnet_pos = 1
    else:
        resnet_pos = None

    sgr_network = SGRNetwork(
        use_semantic=cfg.method.use_semantic,
        include_rgb=cfg.method.include_rgb,
        pretrained_model=(cfg.method.pretrained_model).lower(),
        semantic_downsample_norm=str_none(cfg.method.semantic_downsample_norm),
        semantic_downsample_act=str_none(cfg.method.semantic_downsample_act),
        num_rotation_classes=num_rotation_classes,
        num_grip_classes=2,
        num_collision_classes=2,
        num_points=cfg.method.num_points,
        rand_sample=cfg.method.rand_sample,
        preprocess=cfg.method.preprocess,
        preprocess_norm=str_none(cfg.method.preprocess_norm),
        preprocess_activation=str_none(cfg.method.preprocess_activation),
        proprio_emb_dim=cfg.method.proprio_emb_dim,
        rgb_emb_dim=cfg.method.rgb_emb_dim,
        semantic_dim_per_layer=cfg.method.semantic_dim_per_layer,
        late_proprio_concat=cfg.method.late_proprio_concat,
        resample=cfg.method.resample,
        hierarchical=cfg.method.hierarchical,
        width=cfg_model['encoder_args']['width'] if 'encoder_args' in cfg_model else None,
        resnet_pos=resnet_pos,
        clip_align=cfg.method.clip_align,
        align_type=cfg.method.align_type,
        network_input_image_size=cfg.method.network_input_image_size,
        point_pretrain_apply=cfg.method.point_pretrain.apply,
        point_pretrain_frozen=cfg.method.point_pretrain.frozen,
        point_pretrain_lang=cfg.method.point_pretrain.lang,
        cam_resolution=cam_resolution,
        trans_cls=cfg.method.trans_cls,
        trans_num_classes_per_axis=cfg.method.voxel_sizes[0],
        trans_point_wise=cfg.method.trans_point_wise,
        point_relative_pos=cfg.method.point_relative_pos,
        trans_point_uniform_weight=cfg.method.trans_point_uniform_weight,
        rot_grip_collision_point_wise=cfg.method.rot_grip_collision_point_wise,
        shared_weight=cfg.method.shared_weight,
        rot_grip_collision_shared_weight=cfg.method.rot_grip_collision_shared_weight,
        rot_cls=cfg.method.rot_cls,
        cfg_model=cfg_model,
    )
    agent = SGRAgent(
        layer=0,
        coordinate_bounds=cfg.rlbench.scene_bounds,
        sgr_network=sgr_network,
        camera_names=cfg.rlbench.cameras,
        batch_size=cfg.replay.batch_size,
        voxel_size=cfg.method.voxel_sizes[0],
        num_rotation_classes=num_rotation_classes,
        rotation_resolution=cfg.method.rotation_resolution,
        include_low_dim_state=True,
        lr=cfg.method.lr,
        optimizer_type=cfg.method.optimizer,
        lr_scheduler=cfg.method.lr_scheduler,
        scheduler_type=cfg.method.scheduler_type,
        training_iterations=cfg.framework.training_iterations,
        num_warmup_steps=cfg.method.num_warmup_steps,
        trans_loss_weight=cfg.method.trans_loss_weight,
        rot_loss_weight=cfg.method.rot_loss_weight,
        grip_loss_weight=cfg.method.grip_loss_weight,
        collision_loss_weight=cfg.method.collision_loss_weight,
        lambda_weight_l2=cfg.method.lambda_weight_l2,
        transform_augmentation=cfg.method.transform_augmentation.apply_se3,
        transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
        transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
        transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
        bound_pcd_before_transform=cfg.method.bound_pcd_before_transform,
        trans_cls=cfg.method.trans_cls,
        rot_cls=cfg.method.rot_cls,
        regression_loss=cfg.method.regression_loss,
        color_drop=cfg.method.color_drop,
        feat_drop=cfg.method.feat_drop,
        trans_point_loss=cfg.method.trans_point_loss,
        rot_point_loss=cfg.method.rot_point_loss,
        temperature=cfg.method.temperature,
    )
    agents.append(agent)

    pose_agent = StackAgent(
        agents=agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
        rot_cls=cfg.method.rot_cls,
    )
    preprocess_agent = PreprocessAgent(pose_agent=pose_agent)
    return preprocess_agent
