from dataclasses import dataclass
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def dis_to_cube(pos: np.ndarray, cube_param: np.ndarray) -> float:
    """
    计算点到长方体的距离
    :param pos: 位置向量 (x, y, z)
    :param cube_param: 立方体参数 (x_min, y_min, z_min, x_max, y_max, z_max)
    :return: 点到长方体的距离
    """
    x_min, y_min, z_min, x_max, y_max, z_max = cube_param
    x, y, z = pos
    # 计算每个轴上的距离
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    dz = max(z_min - z, 0, z - z_max)
    # 返回总的距离
    if dx <= 0 and dy <= 0 and dz <= 0:
        return 0.0
    return np.sqrt(dx**2 + dy**2 + dz**2)


def potential_to_cube(pos: np.ndarray, cube_param: np.ndarray, K=1) -> np.ndarray:
    """
    计算长方体在pos处产生的排斥势: K/(1 + d), 1防止除0, 将排斥势控制在K量级
    :param pos: 位置向量 (x, y, z)
    :param cube_param: 立方体参数 (x_min, y_min, z_min, x_max, y_max, z_max)
    :return: 强制转换后的点
    """
    return K / (1 + dis_to_cube(pos, cube_param))


@dataclass
class Building:
    """
    Building class to represent a building in the environment.
    """

    building_id: str
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def to_array(self) -> np.ndarray:
        """
        Convert the building parameters to a numpy array.
        :return: Numpy array of building parameters.
        """
        return np.array(
            [self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z]
        )

    def to_dict(self) -> dict:
        """
        Convert the building parameters to a dictionary.
        :return: Dictionary of building parameters.
        """
        return {
            "building_id": self.building_id,
            "min_x": float(self.min_x),
            "min_y": float(self.min_y),
            "min_z": float(self.min_z),
            "max_x": float(self.max_x),
            "max_y": float(self.max_y),
            "max_z": float(self.max_z),
        }

    def to_vis_json(self) -> dict:
        """
        和前端适配的json格式
        """
        return {
            "id": self.building_id,
            "bbox": [
                self.min_x,
                self.min_y,
                self.min_z,
                self.max_x,
                self.max_y,
                self.max_z,
            ],
        }


def load_buildings(
    filename: str = "harl/envs/pharos_discrete/buildings/train_buildings.csv",
) -> list[Building]:
    """
    Load buildings from a file.
    :param filename: Path to the file containing building data.
    :return: List of Building objects.
    """
    buildings = []
    if not filename.endswith(".csv"):
        raise ValueError("Filename must end with .csv")
    with open(filename, "r") as f:
        # building_id,length,width,min_x,max_x,min_y,max_y,max_z
        data = pd.read_csv(f)
        for _, row in data.iterrows():
            building = Building(
                building_id=int(
                    row["building_id"]
                ),  # HINT: pandas 默认用numpy的int/float, 会序列化出错
                min_x=float(row["min_x"]),
                min_y=0,
                min_z=float(row["min_y"]),  # xyz -> xzy, 其他代码中y是高度
                max_x=float(row["max_x"]),
                max_y=float(row["max_z"]),
                max_z=float(row["max_y"]),
            )
            buildings.append(building)
    return buildings


def plot_building(ax, cube):
    x_min, z_min, y_min, x_max, z_max, y_max = cube
    # 顶点
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]
    )
    # 六个面，每个面由四个顶点索引构成
    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],  # bottom
        [vertices[j] for j in [4, 5, 6, 7]],  # top
        [vertices[j] for j in [0, 1, 5, 4]],  # front
        [vertices[j] for j in [2, 3, 7, 6]],  # back
        [vertices[j] for j in [1, 2, 6, 5]],  # right
        [vertices[j] for j in [4, 7, 3, 0]],  # left
    ]
    poly3d = Poly3DCollection(faces, facecolors="blue", edgecolors="black", alpha=0.5)
    ax.add_collection3d(poly3d)


if __name__ == "__main__":
    # Example usage
    buildings = load_buildings(
        "harl/envs/pharos_discrete/buildings/train_buildings.csv"
    )
    # 绘图
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 画图时用plot_building函数
    for building in buildings:
        cube = building.to_array()
        plot_building(ax, cube)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
