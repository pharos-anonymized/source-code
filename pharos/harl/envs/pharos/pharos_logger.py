import os
from typing import Dict, List, override

import numpy as np
from harl.common.base_logger import BaseLogger
from harl.envs.pharos.pharos_env import PharosEnv
from harl.envs.pharos.utils.render import AgentVis, save_json


class PharosLogger(BaseLogger):
    @override
    def __init__(
        self, args, algo_args, env_args, num_agents, writter, run_dir, envs, eval_envs
    ):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.envs = envs
        self.eval_envs = eval_envs
        if self.env_args["tracing_in_eval"] == True:
            self.vis_dir = self.run_dir + "/vis"
            os.makedirs(self.vis_dir, exist_ok=True)

    @override
    def get_task_name(self):
        return "Pharos"

    @override
    def eval_thread_done(self, tid: int, current_episode: int):
        super().eval_thread_done(tid, current_episode)
        # one_episode_reward_subitem_infos: list(threads) list(steps) list(agents) dict(subitems)
        # -> eval_reward_subitem_infos.append(dict)
        # 对每一个dict key, 求两次list平均(steps, agents)
        dict_keys = self.one_episode_reward_subitem_infos[0][0][0].keys()
        eval_reward_subitem_info = {}  # 用于存储结果的字典
        for k in dict_keys:
            # 提取所有字典中键为k的值，并将它们放入一个列表中
            values = [
                d[k]
                for sublist in self.one_episode_reward_subitem_infos[tid]
                for d in sublist
            ]
            # 计算平均值
            average_value = sum(values) / len(values) if values else 0  # 避免除以零
            eval_reward_subitem_info[k] = average_value  # 将结果存入字典
        self.eval_reward_subitem_infos[tid] = eval_reward_subitem_info
        if self.env_args["tracing_in_eval"]:
            info = self.eval_envs.get_tracing_info(tid)
            # save json
            # print(f"device_info_data len: {len(device_info_data[0])}")
            vis_json_path = os.path.join(
                self.vis_dir, f"vis_{self.episode}_{current_episode}_{tid}.json"
            )
            save_json(
                path=vis_json_path,
                data=info["devices"],
                human_data=info["humans"],
                building_data=info["buildings"],
            )
            self.eval_envs.reset_tracing(tid)

    @override
    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.one_episode_reward_subitem_infos[eval_i].append(eval_infos[eval_i])
        self.eval_infos = eval_infos

    @override
    def eval_log(
        self,
        eval_episode: int,
    ):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        keys = self.eval_reward_subitem_infos[0].keys()
        # save vis json file
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }

        # TODO: attach info to eval_env_info
        for key in keys:
            eval_env_infos[f"eval_avg_{key}s"] = [
                item[key] for item in self.eval_reward_subitem_infos
            ]
        # HINT: attention data shape here
        # calculate avg info item for each item in self.eval_infos
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))

    @override
    def eval_init(self):
        super().eval_init()
        self.one_episode_reward_subitem_infos = []
        self.eval_reward_subitem_infos: List[Dict[str, float]] = []
        if self.env_args["tracing_in_eval"] == True:
            # print(f"Logger enabling tracing in process {os.getpid()}")  # 添加进程ID打印
            for tid in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                self.eval_envs.enable_tracing(tid)
            # print("Logger tracing enabled")

        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_reward_subitem_infos.append([])
            self.eval_reward_subitem_infos.append({})
