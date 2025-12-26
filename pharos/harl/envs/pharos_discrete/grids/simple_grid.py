from queue import Queue
import numpy as np
import random
class Grid3D:
    def __init__(self, x_size: int, y_size: int, z_size: int, num_obstacle_regions=5):
        """
        初始化3D网格并生成障碍物
        
        参数:
        x_size, y_size, z_size -- 网格尺寸
        num_obstacle_regions -- 要生成的障碍物区域数量
        """
        # 保存网格尺寸
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        
        # 创建网格 - 初始值为1(可通过区域)
        self.grid = np.ones((x_size, y_size, z_size), dtype=np.int8)
        
        # 保存原始随机状态
        original_random_state = random.getstate()
        original_numpy_state = np.random.get_state()
        
        # 基于网格形状设置随机种子
        shape_seed = x_size * y_size * z_size
        print(f"shape_seed: {shape_seed}")
        random.seed(shape_seed)
        np.random.seed(shape_seed)
        
        # 生成障碍物
        self._generate_obstacles(num_obstacle_regions)
        
        # 恢复原始随机状态
        random.setstate(original_random_state)
        np.random.set_state(original_numpy_state)
    
    def _generate_obstacles(self, num_regions, max_iter: int=5):
        """生成障碍物区域"""
        for iteration in range(max_iter):
            for _ in range(num_regions):
                # 在xy平面上随机选择一个起始点
                x_center = random.randint(0, self.x_size - 1)
                y_center = random.randint(0, self.y_size - 1)
                
                # 随机确定区域大小
                region_width = random.randint(3, max(3, self.x_size // 5))
                region_height = random.randint(3, max(3, self.y_size // 5))
                
                # 确保区域在网格范围内
                x_min = max(0, x_center - region_width // 2)
                x_max = min(self.x_size, x_center + region_width // 2 + 1)
                y_min = max(0, y_center - region_height // 2)
                y_max = min(self.y_size, y_center + region_height // 2 + 1)
                
                # 障碍物在z轴上的高度，随机选择
                z_height = random.randint(1, max(1, self.z_size // 3))
                
                # 创建障碍物（值设为0表示障碍物）
                self.grid[x_min:x_max, y_min:y_max, 0:z_height] = 0
                
                # 为障碍物添加一些随机性，使边界不那么规则
                self._add_randomness_to_obstacle(x_min, x_max, y_min, y_max, z_height)
            if self.check_connectivity():
                return
        print("Warning: Unable to generate a connected grid after multiple attempts.")
        assert False
        
    def _add_randomness_to_obstacle(self, x_min, x_max, y_min, y_max, z_height):
        """为障碍物边界添加随机性"""
        # 在xy边界上随机添加或移除一些障碍物点
        border_points = []
        
        # 收集边界点
        for x in range(x_min, x_max):
            border_points.append((x, y_min, 0))
            border_points.append((x, y_max-1, 0))
        
        for y in range(y_min, y_max):
            border_points.append((x_min, y, 0))
            border_points.append((x_max-1, y, 0))
        
        # 随机修改边界点
        for _ in range(len(border_points) // 3):  # 修改大约1/3的边界点
            if border_points:
                x, y, z = random.choice(border_points)
                
                # 随机决定是扩展还是收缩障碍物
                if random.random() < 0.7:  # 70%概率扩展障碍物
                    # 向外扩展一格
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.x_size and 0 <= ny < self.y_size and 
                                not (x_min <= nx < x_max and y_min <= ny < y_max)):
                                extend_height = random.randint(1, z_height)
                                self.grid[nx, ny, 0:extend_height] = 0
                else:  # 30%概率收缩障碍物
                    # 这里我们只修改z高度，减少一点高度
                    if z_height > 1:
                        reduce_height = random.randint(1, z_height // 2)
                        new_height = max(1, z_height - reduce_height)
                        if x_min <= x < x_max and y_min <= y < y_max:
                            self.grid[x, y, new_height:z_height] = 1
    
    def get_random_free_point(self):
        """在非障碍物区域随机选取一个点"""
        # 获取所有非障碍物点的索引
        free_points = np.where(self.grid == 1)
        
        # 检查是否有自由点
        if len(free_points[0]) == 0:
            return None
        
        # 随机选择一个索引
        idx = random.randint(0, len(free_points[0]) - 1)
        
        # 返回选中的点坐标
        return (free_points[0][idx], free_points[1][idx], free_points[2][idx])
    
    def is_obstacle(self, x, y, z):
        """检查指定位置是否为障碍物"""
        if 0 <= x < self.x_size and 0 <= y < self.y_size and 0 <= z < self.z_size:
            return self.grid[x, y, z] == 0
        return True  # 网格外的区域视为障碍物
    
    def get_grid(self):
        """返回网格数据"""
        return self.grid
    
    def visualize_xy_slice(self, z_level=0):
        """可视化xy平面上的一个切片"""
        if 0 <= z_level < self.z_size:
            # 使用'#'表示障碍物，'.'表示空白
            for y in range(self.y_size):
                row = ''
                for x in range(self.x_size):
                    row += '#' if self.grid[x, y, z_level] == 0 else '.'
                print(row)
        else:
            print("Z level out of range")
            
    
    def save_json(self):
        pass
    def load_json(self):
        pass
    
    def check_connectivity(self):
        """检查空间是否连通，使用BFS"""
        # 找到第一个非障碍物点作为起点
        start_point = self.get_random_free_point()
        
        if not start_point:
            return True  # 如果没有非障碍物点，认为是连通的
        
        # 使用BFS遍历所有可达点
        visited = np.zeros_like(self.grid, dtype=bool)
        queue = Queue()
        queue.put(start_point)
        visited[start_point] = True
        count_visited = 1
        
        # 计算所有非障碍物点的数量
        free_cells = np.sum(self.grid == 1)
        
        # 定义6个方向的移动（上、下、左、右、前、后）
        directions = [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]
        
        while not queue.empty():
            x, y, z = queue.get()
            
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < self.x_size and 0 <= ny < self.y_size and 0 <= nz < self.z_size and 
                    self.grid[nx, ny, nz] == 1 and not visited[nx, ny, nz]):
                    visited[nx, ny, nz] = True
                    queue.put((nx, ny, nz))
                    count_visited += 1
        
        # 如果访问的点数等于所有自由点数，则空间是连通的
        return count_visited == free_cells