#!/bin/bash

# 配置参数
file_path=../data/searchr1
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# 定义端口列表
ports=(8000 8001 8002 8003 8004 8005 8006 8007)

# 存储进程ID的数组
pids=()

# 信号处理函数，用于清理所有子进程
cleanup() {
    echo
    echo "正在关闭所有检索服务进程..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "关闭进程 $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # 等待进程退出
    sleep 2
    
    # 强制杀死仍在运行的进程
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "强制关闭进程 $pid"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    echo "所有检索服务已关闭"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

echo "正在启动多个检索服务..."

# 启动多个服务进程
for port in "${ports[@]}"; do
    echo "启动检索服务，端口: $port"
    
    python ./retrieval_server.py --index_path $index_file \
                                --corpus_path $corpus_file \
                                --topk 3 \
                                --port $port \
                                --retriever_name $retriever_name \
                                --retriever_model $retriever_path \
                                --faiss_gpu &
    
    # 记录进程ID
    pid=$!
    pids+=($pid)
    echo "进程 $pid 已启动，监听端口 $port"
    
    # 等待一下，避免启动过快
    sleep 2
done

# 获取本机IP地址
local_ip=$(hostname -I | awk '{print $1}')

echo $local_ip > ./retrieval_server_ip.txt

# replace <your server ip> in function_call.py with $local_ip
sed -i "s/<your server ip>/$local_ip/g" ../scripts/exps/asl/asl/envs/retrieval/function_call.py

# 生成服务地址列表
echo
echo "所有检索服务已启动完成！"
echo "服务地址列表（Python列表格式）："
echo -n "["
for i in "${!ports[@]}"; do
    if [ $i -gt 0 ]; then
        echo -n ", "
    fi
    echo -n "\"http://${local_ip}:${ports[i]}\""
done
echo "]"

echo
echo "进程ID列表: ${pids[*]}"
echo "按 Ctrl+C 可以一键关闭所有服务"
echo

# 等待所有子进程
wait
