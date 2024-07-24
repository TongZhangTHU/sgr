import os
import gc
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlbench import ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from agents import peract_bc
from agents import sgr
from agents import semantic_bc_lang
from helpers.utils import setup_logger


def run_seed(rank, cfg: DictConfig, obs_config: ObservationConfig, cams,
             multi_task, seed, world_size) -> None:
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        setup_logger(file_name='train.log',
                     logging_level=cfg.framework.logging_level)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    task_folder = task if not multi_task else tasks
    replay_path = os.path.join(cfg.replay.path, str(task_folder),
                               f'{cfg.method.name}_{cfg.model.name}',
                               cfg.method.tag, 'seed%d' % seed)

    if cfg.method.name == 'PERACT_BC':
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            replay_path if cfg.replay.use_disk else None, cams,
            cfg.method.voxel_sizes, cfg.rlbench.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
            cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes,
            cfg.method.bounds_offset,
            cfg.method.rotation_resolution,
            cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method,
            variation_number=cfg.rlbench.variation_number)

        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'SGR':
        replay_buffer = sgr.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            replay_path if cfg.replay.use_disk else None, cams,
            cfg.method.voxel_sizes, cfg.rlbench.camera_resolution)

        sgr.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
            cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes,
            cfg.method.rotation_resolution,
            keypoint_method=cfg.method.keypoint_method,
            variation_number=cfg.rlbench.variation_number)

        agent = sgr.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'SEMANTIC_BC_LANG':
        replay_buffer = semantic_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            replay_path if cfg.replay.use_disk else None, cams,
            cfg.method.voxel_sizes, cfg.rlbench.camera_resolution)

        semantic_bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
            cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes,
            cfg.method.rotation_resolution,
            keypoint_method=cfg.method.keypoint_method,
            variation_number=cfg.rlbench.variation_number)
        agent = semantic_bc_lang.launch_utils.create_agent(cfg)

    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer,
                                         num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'weights')
    logdir = cwd

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size)

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()