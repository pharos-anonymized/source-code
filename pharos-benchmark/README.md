# Pharos Benchmark

## 🔬 Introduction

This benchmark compares Pharos against two baseline methods for path planning and space allocation:

- **Pharos (Our Method)**: A reinforcement learning-based approach for dynamic space allocation
- **A\* Algorithm**: Classical graph-based pathfinding algorithm
- **Pyomo with IPOPT**: Mathematical optimization solver using the Pyomo modeling framework with IPOPT backend

The benchmark evaluates both path efficiency and computational performance across different scenarios.

## 📦 Installation

### Prerequisites

- Python 3.13+
- IPOPT solver (backend for Pyomo optimization)

### Installation Steps

1. Clone this repository locally:
    ```bash
    git clone <repo-url>
    cd pharos-benchmark
    ```

2. It's recommended to use a virtual environment. You can create one using `conda`:
    ```bash
    conda create -n pharos-benchmark python=3.13
    conda activate pharos-benchmark
    ```

3. Install the project and its dependencies:
    ```bash
    pip install -e .
    ```
    This will install the project in editable mode along with all required dependencies specified in `pyproject.toml`.

4. Install IPOPT solver (required for Pyomo optimization):
    ```bash
    conda install -c conda-forge ipopt
    ```
    **Note**: This project uses Pyomo as the optimization modeling framework with IPOPT as the underlying solver. Alternatively, you can download IPOPT from the [COIN-OR official website](https://coin-or.github.io/Ipopt/).

## 🚀 Usage

**Note:** The files in `data/*.json` contain the initial data for each experiment, including the initial positions of devices, humans, and the building layout. These files are randomly generated using the environment in the main Pharos repository.

### Quick Start

Run all algorithms on multiple agent configurations to see their comparative performance:

```bash
for agents in 10 20 30; do
    for algo in solver astar pharos; do
        echo "Running $algo with $agents agents..."
        python src/scripts/run_algorithm.py $algo data/agents${agents}_humans10_seed0.json --log-level=ERROR
    done
done
```

**Compare reward performance between Pharos and IPOPT solver:**
```bash
python src/scripts/compare_rewards.py \
    output/agents10_humans10_seed0/pharos/rewards.json \
    output/agents10_humans10_seed0/ipopt_continuous_seed42/rewards.json \
    --algo-label="Ipopt"
```

**Analyze human fear across different scenarios:**
```bash
for agents in 10 20 30; do python src/scripts/compare_human_fear.py output/agents${agents}_humans10_seed0; done
```

**Compare entropy across different scenarios:**
```bash
for agents in 10 20 30; do python src/scripts/compare_entropy.py output/agents${agents}_humans10_seed0; done
```

### Command Options

Each algorithm supports various options. Use `--help` to see all available parameters:

```bash
# See all options for each algorithm
python src/scripts/run_algorithm.py solver --help
python src/scripts/run_algorithm.py astar --help
python src/scripts/run_algorithm.py pharos --help

# See comparison tool options
python src/scripts/compare_rewards.py --help
python src/scripts/compare_human_fear.py --help
python src/scripts/compare_entropy.py --help
```

### Individual Algorithm Usage

Run specific algorithms with custom parameters:

```bash
# Run Pyomo solver with IPOPT backend
python src/scripts/run_algorithm.py solver data/agents10_humans10_seed0.json --seed=42

# Run A* path planning algorithm
python src/scripts/run_algorithm.py astar data/agents10_humans10_seed0.json --seed=42

# Run Pharos RL model
python src/scripts/run_algorithm.py pharos data/agents10_humans10_seed0.json --seed=42
```

## 📈 Results and Analysis

### Output Structure

Results are organized by data file and algorithm:

```
output/
├── {data_file_name}/
│   ├── ipopt_continuous_seed{seed}/    # Pyomo solver ipopt backend
│   ├── astar_seed{seed}/               # A* path planning
│   └── pharos/                         # Pharos model
└── figures/                            # Comparison visualizations
```

### Result Files

Each algorithm run generates:
- **`rewards.json`**: Detailed reward breakdown and metrics
- **`times.json`**: Execution time analysis
- **`vis_data.json`**: Complete simulation data for visualization

### Performance Comparison Tools

#### Reward Comparison

```bash
python src/scripts/compare_rewards.py PHAROS_DATA_PATH SOLVER_DATA_PATH
```

Compare reward of Pharos model and Pyomo solver

```bash
python src/scripts/compare_rewards.py \
    output/agents10_humans10_seed0/pharos/rewards.json \
    output/agents10_humans10_seed0/ipopt_continuous_seed42/rewards.json
```

#### Human Fear Comparison

Analyze social metrics and human comfort levels across algorithms.

```bash
python src/scripts/compare_human_fear.py
```

#### Entropy Comparison

Analyze agent spatial distribution entropy across different algorithms. This tool measures how evenly agents are distributed across the space, with higher entropy indicating more uniform distribution.

```bash
python src/scripts/compare_entropy.py EXPERIMENT_DIR
```

**Example usage:**
```bash
python src/scripts/compare_entropy.py output/agents10_humans10_seed0
```

## ⚙️ Configuration

### Environment Settings

```python
# World boundaries
world_bounds: [30.0, 10.0, 30.0]

# Reward weights
closer_factor: 10.0          # Goal approach reward
reach_factor: 30.0           # Goal completion bonus
scare_factor: 25.0           # Human discomfort penalty
collision_factor: 1000.0     # Collision penalty
cutoff_scare_distance: 5.0   # Social distance threshold

# Agent constraints
agent_max_speed: 10.0        # Maximum agent velocity

# Human behavior model
human_speed_mean: 1.0        # Average human walking speed
human_speed_std: 0.5         # Speed variation
human_velocity_update_interval: 100.0  # Behavior update frequency
```

### Algorithm-Specific Parameters

#### Pyomo Solver
- **Tolerance**: 1e-6
- **Max iterations**: 500
- **Linear solver**: MUMPS

#### A* Path Planning
- **Movement directions**: 6-directional movement
- **Heuristic**: Manhattan distance
- **Collision threshold**: 0.5 units

## 🏗️ Project Structure

```
pharos-benchmark/
├── src/
│   ├── scripts/             # Command-line scripts
│   │   ├── run_algorithm.py     # Unified algorithm runner
│   │   ├── compare_rewards.py   # Reward comparison tool
│   │   ├── compare_entropy.py   # Entropy analysis tool
│   │   └── compare_human_fear.py # Human fear analysis tool
│   ├── runner/              # Algorithm runners
│   │   ├── base_runner.py       # Base runner class
│   │   ├── pharos_runner.py     # Pharos RL runner
│   │   ├── solver_runner.py     # Pyomo solver runner
│   │   └── astar_runner.py      # A* algorithm runner
│   ├── algorithms/          # Algorithm implementations
│   │   ├── astar.py             # A* path planning implementation
│   │   ├── rl_model.py          # RL model implementation
│   │   └── pyomo_solver.py      # Pyomo optimization modeling core
│   └── core/                # Core utilities and environment
│       ├── env.py               # Environment simulation
│       ├── log_data.py          # Data management utilities
│       └── utils.py             # General utilities
├── data/                    # Benchmark data files
├── output/                  # Generated results
├── requirements.txt         # Dependencies
├── pyproject.toml           # Project configuration
└── README.md               # Project documentation
```

## 🔧 Algorithm Command Reference

### Unified Command Structure

All algorithms now use the same command structure:

```bash
python src/scripts/run_algorithm {algorithm} {data_file} [options]
```

Where `{algorithm}` is one of:
- `solver` - Pyomo solver with IPOPT backend
- `astar` - A* path planning algorithm  
- `pharos` - Pharos RL model

### Common Options

- `--seed`: Random seed for reproducibility
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--help`: Show algorithm-specific help

### Examples

```bash
# Run all algorithms on the same dataset
python src/scripts/run_algorithm solver data/agents10_humans10_seed0.json
python src/scripts/run_algorithm astar data/agents10_humans10_seed0.json  
python src/scripts/run_algorithm pharos data/agents10_humans10_seed0.json

# Run with specific seed and reduced logging
python src/scripts/run_algorithm solver data/agents10_humans10_seed0.json --seed=42 --log-level=ERROR
```
