import json
import os

import numpy as np
import torch
from gym import spaces
from tqdm import tqdm

from harl.envs.pharos.pharos_env import PharosEnv
from harl.algorithms.actors import HAPPO
class MAPPOEvaluator:
    def __init__(self, actor_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.actor = actor_model
        self.device = device
        self.actor.eval()

        # 获取RNN配置
        self.use_rnn = (hasattr(self.actor, '_use_naive_recurrent_policy') and
                        self.actor._use_naive_recurrent_policy) or \
                       (hasattr(self.actor, '_use_recurrent_policy') and
                        self.actor._use_recurrent_policy)

        if self.use_rnn:
            self.hidden_size = self.actor.rnn.hidden_size
            self.rnn_layers = getattr(self.actor.rnn, 'num_layers', 1)

    def init_rnn_states(self, batch_size=1):
        """初始化RNN状态"""
        if self.use_rnn:
            # 创建正确维度的RNN状态: [num_layers, batch_size, hidden_size]
            return torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device)
        return torch.zeros(1, batch_size, 1).to(self.device)  # 返回dummy状态而不是None

    @torch.no_grad()
    def evaluate(self, obs, rnn_states=None, episode_done=False):
        """
        进行单步评估
        :param obs: 观察值 (numpy array 或 torch.Tensor)
        :param rnn_states: RNN隐藏状态
        :param episode_done: 当前episode是否结束
        """
        # 确保obs是tensor并且在正确的设备上
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)

        # 添加batch维度如果需要
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 初始化或验证RNN状态
        if rnn_states is None:
            rnn_states = self.init_rnn_states()
        elif isinstance(rnn_states, np.ndarray):
            rnn_states = torch.FloatTensor(rnn_states).to(self.device)

        # 创建masks
        masks = torch.zeros(1, 1) if episode_done else torch.ones(1, 1)
        masks = masks.to(self.device)

        # 执行前向传播
        actions, _, new_rnn_states = self.actor(
            obs,
            rnn_states,
            masks,
            available_actions=None,
            deterministic=True
        )

        # 转换动作为numpy数组（如果需要）
        actions = actions.squeeze().cpu().numpy()

        return actions, new_rnn_states

if __name__ == '__main__':
    # 评测 + 保存图像
    run_dir = "results/pharos/pharos/happo/test/seed-00042-2025-03-26-02-06-47"
    model_dir = f"{run_dir}/models"
    algo_args = json.load(open(f"{run_dir}/config.json"))["algo_args"]["algo"]
    eval_episodes = 100
    eval_episodes_timesteps = 50
    env = PharosEnv(args={})
    agent_num = env.n_agents
    actor_models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(agent_num):
        checkpoint = torch.load(model_dir + f'/actor_agent{i}.pt', map_location=device)
        print(checkpoint)
        model = HAPPO(args=algo_args, obs_space=env.observation_space,
                      act_space=env.action_space,
                      device=device)
        model.load_state_dict(checkpoint)

        model.eval()
        actor_models.append(model)
    print(actor_models[0])
    evaluators = [MAPPOEvaluator(actor_model) for actor_model in actor_models]
    env.seed(0)
    env.reset()


    all_rewards = []
    all_infos = []
    log_interval = 10
    pos_vec_list = []
    for i in tqdm(range(eval_episodes)):
        env.reset()
        rewards = []
        infos: list[dict[str, float]] = []
        rnn_states: list = [evaluator.init_rnn_states() for evaluator in evaluators]
        for j in range(eval_episodes_timesteps):
            actions = []

        reward_by_episode = np.sum(np.array(rewards))

        # calculate average info items
        info_by_episode = {}
        for k in infos[0].keys():
            info_by_episode[k] = sum([info[k] for info in infos])
        if i % log_interval == 0:
            print(f"Episode {i} safe time reward: {info_by_episode['safe_time_reward']}")
            print(f"Episode {i} reward: {reward_by_episode}")
            print(f"Episode {i} info: {info_by_episode}")
        all_rewards.append(reward_by_episode)
        all_infos.append(info_by_episode)

    print(f"Average reward: {sum(all_rewards) / eval_episodes}")
    for k in all_infos[0].keys():
        print(f"Average {k}: {sum([info[k] for info in all_infos]) / eval_episodes}")
    if len(pos_vec_list) != 0:
        print(f"pos_vec_list length: {len(pos_vec_list)}")
        np.save("eval_pos_vecs.npy", np.array(pos_vec_list))
    print(f"重叠的比率为: {len([info['overlap_penalty'] for info in all_infos if info['overlap_penalty'] > 0.1]) / eval_episodes}")
