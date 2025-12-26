#!/usr/bin/env python3
import argparse
import subprocess
import datetime
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="启动训练任务脚本")
    parser.add_argument(
        "-i", type=str, help="指定agent数量，多个用逗号分隔，如：-i 2,4,6"
    )
    parser.add_argument(
        "-a",
        type=str,
        choices=["mappo", "hasac", "hatrpo", "happo"],
        help="指定算法，默认为mappo",
    )
    # 由于--k v参数不固定，使用parse_known_args捕获未知参数
    args, unknown = parser.parse_known_args()

    # 解析 --k v 参数对
    kv_args = []
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i]
            if i + 1 >= len(unknown):
                print(f"参数 {key} 缺少对应的值")
                sys.exit(1)
            value = unknown[i + 1]
            kv_args.append(key)
            kv_args.append(value)
            i += 2
        else:
            print(f"未知参数格式: {unknown[i]}")
            sys.exit(1)

    return args, kv_args


def interactive_input():
    agents_nums = []
    print("请输入agents数量，回车确认，输入-1结束：")
    while True:
        num = input().strip()
        if num == "-1":
            break
        if num.isdigit():
            agents_nums.append(num)
        else:
            print("请输入有效的正整数或-1")

    kv_args = []
    kv_suffix = ""
    add_kv = input("是否需要添加额外的--k v参数？(y/n) ").strip().lower()
    if add_kv == "y":
        while True:
            k = input("请输入参数名（不带--，直接回车结束）：").strip()
            if k == "":
                break
            v = input("请输入参数值：").strip()
            kv_args.append(f"--{k}")
            kv_args.append(v)
    return agents_nums, kv_args


def main():
    args, kv_args = parse_args()

    if args.i:
        agents_nums = args.i.split(",")
        # 简单校验数字
        for n in agents_nums:
            if not n.isdigit():
                print(f"无效的agent数量: {n}")
                sys.exit(1)
    algo = "mappo"  # 默认算法
    if args.a:
        algo = args.a
    else:
        agents_nums, kv_args_interactive = interactive_input()
        kv_args.extend(kv_args_interactive)

    # 构造kv_suffix用于日志文件名
    kv_suffix = ""
    for i in range(0, len(kv_args), 2):
        k = kv_args[i][2:]  # 去掉前缀 --
        v = kv_args[i + 1]
        kv_suffix += f"-{k}={v}"

    date_str = datetime.datetime.now().strftime("%m-%d")

    # 激活conda环境的命令在python中不生效，建议用户提前激活环境或用shell脚本调用此脚本
    # 这里仅打印提示
    print("请确保已激活 conda 环境 pharos")

    for num in agents_nums:
        print(f"启动任务：MA={num}")
        log_file = f"train-{date_str}-{algo}-{num}A{kv_suffix}.log"
        print(f"日志文件：{log_file}")

        cmd = [
            "python",
            "examples/train.py",
            "--algo",
            algo,
            "--env",
            "pharos_discrete",
            "--N_Agents",
            num,
            "--num_env_steps",
            "5000000",
            "--n_rollout_threads",
            "100",
            "--n_eval_rollout_threads",
            "8",
            "--activation_func",
            "silu",
        ] + kv_args

        print("命令:", " ".join(cmd))
        with open(log_file, "w") as f:
            # 启动子进程并等待完成
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"任务pid: {process.pid}")
            process.wait()
            print(f"任务MA={num} 完成，启动下一个任务...")

    print("所有任务完成")


if __name__ == "__main__":
    main()
