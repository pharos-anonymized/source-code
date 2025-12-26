#!/bin/zsh

agents_nums=()

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

date_str=$(date +%m-%d)

for num in $agents_nums; do
  log_file="test-${date_str}-mappo-${num}A.log"
  echo "启动测试任务：MA=${num}，日志文件: $log_file"

  nohup zsh -c "echo '任务 MA=$num 开始'; sleep $((num)); echo '任务 MA=$num 结束'" > "$log_file" 2>&1 &

  pid=$!
  echo "测试任务pid: $pid"

  wait $pid
  echo "测试任务 MA=$num 完成，启动下一个..."
done

echo "所有测试任务完成"