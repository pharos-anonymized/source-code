# 写一个程序, 在(50, 50, 10)的离散范围内不断随机生成起始点和目标点, 只允许在6个方向上移动, 不允许斜行
# 计算平均而言, 起始点到目标点需要多少次拐弯

import numpy as np

def generate_random_point():
    """生成随机点 (x, y, z)"""
    return np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100)


def calculate_straight_moves(start, target):
    """
    计算从起始点到目标点的路径中需要的直线移动次数。
    每次直线移动可以移动1~10格，只允许在6个方向上移动，不允许斜行。
    尽可能最小化直线移动次数。
    """
    import numpy as np

    current = list(start)
    moves = 0

    # 按照x, y, z的顺序依次处理每个方向
    for dim in range(3):
        # 计算当前维度上需要移动的距离
        distance = target[dim] - current[dim]

        if distance == 0:
            continue

        # 确定移动方向
        direction = np.sign(distance)
        # 计算需要移动的绝对距离
        abs_distance = abs(distance)

        # 计算需要的完整移动次数（每次最多移动10格）
        full_moves = abs_distance // 10
        # 计算剩余的距离（不足10格的部分）
        remainder = abs_distance % 10

        # 更新当前位置，先处理完整的10格移动
        current[dim] += direction * (full_moves * 10)
        moves += full_moves

        # 如果有剩余距离，再增加一次移动
        if remainder > 0:
            current[dim] += direction * remainder
            moves += 1

    return moves

def simulate(num_trials=1000):
    """模拟多次随机生成起始点和目标点，计算平均拐弯次数"""
    total_turns = 0
    for _ in range(num_trials):
        start = generate_random_point()
        target = generate_random_point()
        while start == target:  # 确保起始点和目标点不同
            target = generate_random_point()
        total_turns += calculate_straight_moves(start, target)
    return total_turns / num_trials

if __name__ == "__main__":
    avg_turns = simulate(1000)
    print(f"串行最大理论奖励: {1 / avg_turns * 100}")
    N = 10
    print(f"最大理论奖励: {N / avg_turns * 100}") # 88
    # 目前寻出来的: 72
    print(f"平均移动时间: {avg_turns:.2f}")