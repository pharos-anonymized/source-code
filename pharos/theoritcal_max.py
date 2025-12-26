import random
import math


def random_point():
    # 在 (30,10,30) 三维格子中随机选取一个点，假设点坐标为整数
    x = random.randint(0, 29)
    y = random.randint(0, 9)
    z = random.randint(0, 29)
    return (x, y, z)


def euclidean_distance(p1, p2):
    # 计算两点之间的欧几里得距离
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def direction_cosines(p1, p2):
    # 计算两点连线方向的单位向量
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0:
        # 两点相同返回0余弦，避免除零
        return (0, 0, 0)
    # 方向余弦是单位向量的三个分量
    return (dx / length, dy / length, dz / length)


def max_cosine_of_direction(p1, p2):
    cosines = direction_cosines(p1, p2)
    # 三个方向的余弦值绝对值
    abs_cosines = [abs(c) for c in cosines]
    return max(abs_cosines)


def manhanton_distance(p1, p2):
    # 曼哈顿距离
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


def main(samples=10000):
    max_cos_values = []
    man_distances = []
    eu_distances = []
    for _ in range(samples):
        p1 = random_point()
        p2 = random_point()
        # 避免起点和终点相同导致的异常
        if p1 == p2:
            continue
        max_cos = max_cosine_of_direction(p1, p2)
        max_cos_values.append(max_cos)
        man_distances.append(manhanton_distance(p1, p2))
        eu_distances.append(euclidean_distance(p1, p2))
    average_max_cos = sum(max_cos_values) / len(max_cos_values)
    average_distance = sum(man_distances) / len(man_distances)
    # manhanton_distance 步, distance距离closer, 1 target达到
    avg_rewards = [(e + 1 * 30 / 10) / d for e, d in zip(eu_distances, man_distances)]
    average_reward = sum(avg_rewards) / len(avg_rewards)
    print(
        f"随机采样 {len(max_cos_values)} 对点，最大方向余弦值的平均值为: {average_max_cos:.4f}"
    )
    print(
        f"随机采样 {len(man_distances)} 对点，曼哈顿距离的平均值为: {average_distance:.4f}"
    )
    print(f"随机采样 {len(avg_rewards)} 对点，平均奖励值为: {average_reward:.4f}")


if __name__ == "__main__":
    main()
    # 0.87-0.88
