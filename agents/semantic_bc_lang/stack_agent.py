import torch
import numpy as np
from typing import List
from helpers import utils
from yarr.agents.agent import Agent, ActResult, Summary
from agents.semantic_bc_lang.semantic_bc_lang_agent import SemanticBCLangAgent


class StackAgent(Agent):

    def __init__(self, qattention_agents: List[SemanticBCLangAgent],
                 rotation_resolution: float, camera_names: List[str]):
        super(StackAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        total_losses = 0.
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            replay_sample.update(update_dict)
            total_losses += update_dict['total_loss']
        return {
            'total_losses': total_losses,
        }

    def act(self,
            step: int,
            observation: dict,
            deterministic=False) -> ActResult:

        observation_elements = {}
        translation_results, rot_grip_results, ignore_collisions_results = [], [], []
        infos = {}
        for depth, qagent in enumerate(self._qattention_agents):
            act_results = qagent.act(step, observation, deterministic)
            attention_coordinate = act_results.observation_elements[
                'attention_coordinate'].cpu().numpy()
            observation_elements['attention_coordinate_layer_%d' %
                                 depth] = attention_coordinate[0]

            translation_idxs, rot_grip_idxs, ignore_collisions_idxs = act_results.action
            translation_results.append(translation_idxs)
            if rot_grip_idxs is not None:
                rot_grip_results.append(rot_grip_idxs)
            if ignore_collisions_idxs is not None:
                ignore_collisions_results.append(ignore_collisions_idxs)

            observation['attention_coordinate'] = act_results.observation_elements['attention_coordinate']

            for n in self._camera_names:
                px, py = utils.point_to_pixel_index(
                    attention_coordinate[0],
                    observation['%s_camera_extrinsics' % n][0, 0].cpu().numpy(),
                    observation['%s_camera_intrinsics' % n][0, 0].cpu().numpy())
                pc_t = torch.tensor([[[py, px]]],
                                    dtype=torch.float32,
                                    device=self._device)
                observation['%s_pixel_coord' % n] = pc_t
                observation_elements['%s_pixel_coord' % n] = [py, px]

            infos.update(act_results.info)

        rgai = torch.cat(rot_grip_results, 1)[0].cpu().numpy()
        ignore_collisions = float(
            torch.cat(ignore_collisions_results, 1)[0].cpu().numpy())
        if self._qattention_agents[0]._trans_cls:
            observation_elements['trans_action_indicies'] = torch.cat(
                translation_results, 1)[0].cpu().numpy()
        else:
            observation_elements['trans_action_indicies'] = None
        observation_elements['rot_grip_action_indicies'] = rgai
        continuous_action = np.concatenate([
            act_results.observation_elements['attention_coordinate'].cpu().
            numpy()[0],
            utils.discrete_euler_to_quaternion(rgai[-4:-1],
                                               self._rotation_resolution),
            rgai[-1:],
            [ignore_collisions],
        ])
        return ActResult(continuous_action,
                         observation_elements=observation_elements,
                         info=infos)

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
