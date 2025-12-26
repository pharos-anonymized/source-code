# Pharos

## Installation

```bash
conda create -n pharos python=3.12
conda activate pharos
# Manually install PyTorch (tested: 2.7.1+cu126)
git clone --recurse-submodules <project_url>
cd <project>
pip install -e .
```

## Training

### Batch Training

To train with multiple agent counts (4, 6, 8, 10, 15, 20), use:

```bash
./train_diff_agents.sh -i 4,6,8,10,15,20 &
```

Logs are saved as `train-<date>-mappo-<agent_num>A.log`.

To add training parameters, use: `./train_diff_agents.sh --k v`  
Example: `--enable_human_fear True --enable_buildings True`

- Environment parameters: `harl/configs/envs_cfgs/pharos_discrete.yaml`
- Algorithm parameters: `harl/configs/algos_cfgs/<your_algo>.yaml` (e.g., `mappo.yaml`)

**Training time:**  
With default settings and a single machine (Nvidia A6000, AMD EPYC 9554), approximate times are: 1003s, 1454s, 1981s, and 2535s for increasing agent counts.

### Continue Training

Resume training from a saved model using `--model_dir`:

```bash
python examples/train.py \
  --algo mappo \
  --env pharos_discrete \
  --N_Agents 10 \
  --num_env_steps 5000000 \
  --n_rollout_threads 56 \
  --n_eval_rollout_threads 8 \
  --model_dir results/pharos_discrete/pharos_discrete/mappo/installtest/seed-00001-2025-06-27-16-11-02/models
```

### Training File Structure

Training logs and outputs are stored in:  
`<logger_dir>/<env_name>/<task_name>/<algo_name>/<exp_name>/seed-<seed>-<datetime>`

Example:  
`results/pharos_discrete/pharos_discrete/mappo/installtest/seed-00001-2025-06-27-03-54-45`

The internal directory structure is as follows: 
```
├── config.json
├── logs
├── models
│   ├── actor_agent0.pt
│   ├── ... 
│   ├── critic_agent.pt
│   └── value_normalizer.pt
├── progress.txt
└── vis
    ├── vis_1000_0.json
    ...
```

- `config.json`: Training configuration (`env_args`, `algo_args`)
- `logs`: Training logs (view with `tensorboard --logdir <log_dir>`)
- `models`: Saved model files
- `vis`: Visualization files from evaluation


## Plotting

Run experiments to generate data before plotting.

- **Compare average rewards for different agent counts:**

    ```bash
    python draw_picture.py --plot-type agent-rewards --agent-nums 4 6 8 10 --has-human --has-building --filename avg_agent_rewards_diff_agents.png
    ```
    ![](./assets/avg_agent_rewards_diff_agents.png)

- **Plot subplots of single-experiment metrics:**

    ```bash
    python draw_picture.py --plot-type subgraphs --agent-nums 10 --has-human --has-building --filename metrics_subgraphs.png
    ```

- **Compare rewards for different algorithms:**

    ```bash
    python draw_picture.py --plot-type algo-comparison --agent-nums 10 --has-human --has-building --filename algo_comparison_10A.png
    ```
    ![](./assets/avg_agent_rewards_diff_algos_10A.png)

- **Spatial entropy chart:**

    ```bash
    python entropy.py --base_dir <base_dir> --output_file entropy.png
    ```
    ![](./assets/entropy.png)

## Benchmarking Pharos Against Baseline Algorithms

Pharos has been benchmarked against two widely-used baseline methods:

- **A\* Algorithm**: A classical graph-based pathfinding algorithm, used here as a strong baseline for discrete path planning.
- **Pyomo with IPOPT Solver**: A mathematical optimization approach, modeling the problem as a continuous optimization task and solving it with the IPOPT backend.

These benchmarks assess both **solution quality** (e.g., total reward, human fear, spatial entropy) and **computational performance** (e.g., runtime, scalability) across diverse scenarios and varying agent counts.

### Benchmarking Overview

- **Data**: All experiments use the same initial conditions, with data files specifying agent, human, and building positions. These are generated using the Pharos environment.
- **Algorithms**: Each scenario is run with all three algorithms (Pharos, A*, Pyomo+IPOPT) for direct comparison.
- **Metrics**: The following metrics are collected and compared:
    - **Total reward**: Aggregate performance of agents, including penalties for collisions and human discomfort.
    - **Execution time**: Computational efficiency of each method.
    - **Human fear**: Social comfort metric, measuring how well agents avoid scaring humans.
    - **Spatial entropy**: How evenly agents are distributed in the environment.

### Running the Benchmark

To run or reproduce the benchmarking experiments, please refer to the `pharos-benchmark` repository and its README for setup and instructions.

## Visualization

For 3D visualization of Pharos tracking data, use the `pharos-visual-3d` repository.

**pharos-visual-3d** is a web-based 3D visualization tool built with React and Three.js. It allows you to interactively view, track, and analyze the movement of devices (drones), humans, and buildings in a simulated environment.

### Key Features

- **3D Visualization**: Interactive 3D scene with orbital camera controls.
- **Real-time Tracking**: Visualize devices, humans, and buildings with position and velocity data.
- **Time-based Playback**: Timeline controls for playing back historical movement data.
- **Data Import**: Drag-and-drop JSON file support for loading tracking data.

For more details, data format, and advanced usage, see the README of `pharos-visual-3d` repository.

![](./assets/frontend.png)


## Acknowledgements
The [PKU-HARL project](https://github.com/PKU-MARL/HARL) is the base repository for this repository.
