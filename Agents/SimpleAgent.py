import json
from abc import ABC

from Agents.data import mip_task
import numpy as np

import ray
import gym
import ray.rllib.agents.ppo as ppo
import tensorflow as tf

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict, Optional, TYPE_CHECKING, List, Any, Union, NamedTuple
from ray.rllib.policy import Policy


class Answer(NamedTuple):
    best_g: float
    best_actions: np.ndarray


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float):
            return float(o)
        if isinstance(o, np.int):
            return int(o)
        return super(NpEncoder, self).default(o)


class SampleCallback(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.best_reward = -666666666
        self.best_actions = []
        self.legacy_callbacks = legacy_callbacks_dict or {}

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: Episode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        sample_obj = original_batches[agent_id][1]
        rewards = sample_obj.columns(['rewards'])[0]
        total_reward = np.sum(rewards)
        actions = sample_obj.columns(['actions'])[0]

        if total_reward > self.best_reward and len(actions[rewards >= 0]) >= self.bound:
            actions = actions[rewards >= 0]
            total_reward = np.sum(actions * self.p[:len(actions)])
            self.best_actions = actions
            self.best_reward = total_reward
            episode.hist_data["best_actions"] = [actions]
            episode.hist_data["best_reward"] = [total_reward]


class MyEnv(gym.Env, ABC):
    def __init__(self, env_config: dict):
        self.m = env_config['m']
        self.n = env_config['n']
        self.b = env_config['b']
        self.c = env_config['c']
        self.ubound = env_config['ubound']
        self.p = env_config['p']
        self.action_space = gym.spaces.Discrete(self.ubound + 1)
        self.observation_space = gym.spaces.Dict({
            'rem': gym.spaces.Box(low=np.zeros(self.n), high=self.b, dtype=np.float64),
            'j': gym.spaces.Discrete(self.m + 1), })
        self.state = {'rem': np.array(self.b), 'j': 0}
        self.done = False
        self.reward = 0

    def reset(self, **kwargs):
        self.state = {'rem': np.array(self.b), 'j': 0}
        self.done = False
        return self.state

    def step(self, action):
        # print('current state:', self.state)
        # print('action taken:', action)
        j = self.state['j']
        rem = self.state['rem'] - self.c[:, j] * action
        if np.any(rem < 0):
            self.reward = -1
        else:
            self.reward = action * self.p[j]
            j += 1
            self.state = {'rem': rem, 'j': j}

        # print('reward:', self.reward)
        # print('next state:', self.state)

        if j == self.m:
            self.done = True
            # print(rem)
        else:
            self.done = False

        return self.state, self.reward, self.done, {}


def train(ubound: int, c: np.ndarray, b: np.ndarray, p: np.ndarray, m: int, n: int, bound: int, steps_per_forward: int):
    ray.shutdown()
    ray.init()

    obj = SampleCallback
    obj.bound = bound
    obj.p = p

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "torch"
    config['disable_env_checking'] = True
    config["env_config"] = {'ubound': ubound, 'c': c, 'b': b, 'p': p, 'm': m, 'n': n}
    config['callbacks'] = SampleCallback
    config['lr'] = 3e-4

    model = ppo.PPOTrainer(env=MyEnv, config=config)

    best_g = 0
    best_actions = []

    for i in range(steps_per_forward):
        result = model.train()

        if 'best_reward' in result['hist_stats'] and len(result['hist_stats']['best_reward']) > 0 and (
                best_g < result['hist_stats']['best_reward'][-1] or best_actions == []):
            best_g = result['hist_stats']['best_reward'][-1]
            best_actions = result['hist_stats']['best_actions'][-1]

        if i % 10 == 0:
            print('i: ', i)
            print('mean episode length:', result['episode_len_mean'])
            print('max episode reward:', result['episode_reward_max'])
            print('mean episode reward:', result['episode_reward_mean'])
            print('min episode reward:', result['episode_reward_min'])
            print('total episodes:', result['episodes_total'])
            print('solution:', best_g, best_actions)
        checkpoint = model.save()

    ans = Answer(best_g=best_g, best_actions=best_actions)
    return ans


conclusion = train(
    mip_task.ubound,
    mip_task.constraits,
    mip_task.bounds,
    mip_task.coefs,
    mip_task.m,
    mip_task.n,
    mip_task.r_lenght,
    mip_task.steps_per_forward)

json.dump(conclusion, cls=NpEncoder, fp=open('result.json', 'w+'))
