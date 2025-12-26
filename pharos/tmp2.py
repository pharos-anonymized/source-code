import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
def gen_sample_data(n_samples=1000, input_shape=(20, 20, 20)):
    """
    生成随机三维势场数据

    参数:
        n_samples (int): 样本数量
        input_shape (tuple): 输入数据的形状

    返回:
        随机生成的三维势场数据
    """
    # 在空间中随机生成1-5个源点, 按照-1/r生成势场
    data = np.zeros((n_samples, *input_shape))
    for i in range(n_samples):
        # 随机生成源点位置
        n_sources = np.random.randint(2, 3)  # 1到5个源点
        sources = np.random.rand(n_sources, 3) * (input_shape[0])
        sources = sources.astype(int)

        # 计算势场
        for source in sources:
            x, y, z = np.indices(input_shape)
            r = 0.5 + np.sqrt((x - source[0])**2 + (y - source[1])**2 + (z - source[2])**2)
            data[i] -= 1 / r

    print(np.mean(data))
    print(np.min(data))
    return torch.tensor(data, dtype=torch.float32)

data = gen_sample_data(n_samples=1)
# 计算势场在三个方向上的近似梯度
def compute_gradient(data):
    """
    计算势场在三个方向上的近似梯度

    参数:
        data (torch.Tensor): 输入数据

    返回:
        三个方向上的梯度
    """
    dx = data[:, 1:, :, :] - data[:, :-1, :, :]
    dy = data[:, :, 1:, :] - data[:, :, :-1, :]
    dz = data[:, :, :, 1:] - data[:, :, :, :-1]
    return dx, dy, dz

# 绘制梯度曲线
def plot_gradient(data, title=''):
    """
    绘制梯度曲线

    参数:
        data (torch.Tensor): 输入数据
        title (str): 图像标题
    """
    dx, dy, dz = compute_gradient(data)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(dx[0].flatten(), label='dx')
    axs[1].plot(dy[0].flatten(), label='dy')
    axs[2].plot(dz[0].flatten(), label='dz')
    for ax in axs:
        ax.legend()
        ax.set_xlabel('Index')
        ax.set_ylabel('Gradient Value')
        ax.set_title(title)
    plt.show()
plot_gradient(data, title='Gradient of Potential Field')