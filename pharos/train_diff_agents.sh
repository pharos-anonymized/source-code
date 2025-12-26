#!/bin/zsh

function usage() {
  echo "用法: $0 [-i agents_nums] [--k v ...]"
  echo "  -i 指定agent数量，多个用逗号分隔，如：-i 2,4,6"
  echo "  --k v 可选参数对，可以有多个，如：--seed 42 --lr 0.01"
  echo "如果不使用 -i 参数，则进入交互输入模式，每行输入一个数字，输入-1结束"
  exit 1
}

agents_nums=()
kv_args=()
kv_suffix=""

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      shift
      IFS=',' read -r -A agents_nums <<< "$1"
      shift
      ;;
    --*)
      k="${1#--}"
      v="$2"
      kv_args+=("--$k" "$v")
      kv_suffix="${kv_suffix}-${k}=${v}"
      shift 2
      ;;
    -*)
      usage
      ;;
    *)
      shift
      ;;
  esac
done

# 如果没有通过-i传参，进入交互输入模式
if [[ ${#agents_nums[@]} -eq 0 ]]; then
  echo "请输入agents数量，回车确认，输入-1结束："
  while true; do
    read num
    if [[ "$num" == "-1" ]]; then
      break
    fi
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      agents_nums+=($num)
    else
      echo "请输入有效的正整数或-1"
    fi
  done

  # 交互式输入kv参数
  echo "是否需要添加额外的--k v参数？(y/n)"
  read add_kv
  if [[ "$add_kv" == "y" ]]; then
    while true; do
      echo "请输入参数名（不带--，直接回车结束）："
      read k
      if [[ -z "$k" ]]; then
        break
      fi
      echo "请输入参数值："
      v_sanitized="${v//\//-}"
      kv_args+=("--$k" "$v")
      kv_suffix="${kv_suffix}-${k}=${v_sanitized}"
    done
  fi
fi

# 先激活环境（如果需要）
conda activate pharos

date_str=$(date +%m-%d)

for num in $agents_nums; do
  echo "启动任务：MA=$num"

  log_file="train-${date_str}-mappo-${num}A${kv_suffix}.log"
  echo "日志文件：$log_file"
  echo "${kv_args[@]} > $log_file"
  nohup python examples/train.py --algo mappo --env pharos_discrete --N_Agents $num --num_env_steps 5000000 --n_rollout_threads 100 --n_eval_rollout_threads 8 "${kv_args[@]}" > "$log_file" 2>&1 &

  pid=$!
  echo "任务pid: $pid"

  wait $pid
  echo "任务MA=$num 完成，启动下一个任务..."
done

echo "所有任务完成"