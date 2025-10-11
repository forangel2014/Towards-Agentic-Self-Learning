import os
import pandas as pd
import argparse
from pathlib import Path
import json
import jsonlines

def modify_reward_model_style(df):
    """
    修改DataFrame中所有样本的[reward_model][style]字段为"rule"
    
    Args:
        df (pd.DataFrame): 输入的DataFrame
        
    Returns:
        pd.DataFrame: 修改后的DataFrame
    """
    modified_df = df.copy()
    
    # 检查是否存在reward_model列
    if 'reward_model' in modified_df.columns:
        print("发现reward_model列，正在修改style字段...")
        
        def update_reward_model(reward_model_data):
            """更新reward_model数据中的style字段"""
            if pd.isna(reward_model_data):
                return reward_model_data
            
            try:
                # 如果是字符串，尝试解析为JSON
                if isinstance(reward_model_data, str):
                    data = json.loads(reward_model_data)
                else:
                    data = reward_model_data
                
                # 修改style字段
                if isinstance(data, dict):
                    data['style'] = 'rule'
                    return json.dumps(data) if isinstance(reward_model_data, str) else data
                else:
                    return reward_model_data
                    
            except (json.JSONDecodeError, TypeError, AttributeError):
                # 如果解析失败，返回原始数据
                print(f"警告: 无法解析reward_model数据: {reward_model_data}")
                return reward_model_data
        
        # 应用修改
        modified_df['reward_model'] = modified_df['reward_model'].apply(update_reward_model)
        print("已将所有样本的[reward_model][style]字段改为'rule'")
    else:
        print("警告: 未找到reward_model列")
    
    return modified_df

def read_data_file(file_path):
    """
    读取数据文件，支持parquet和jsonl格式
    
    Args:
        file_path (Path): 文件路径
        
    Returns:
        pd.DataFrame: 读取的数据
    """
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix == '.jsonl':
        data_list = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data_list.append(obj)
        return pd.DataFrame(data_list)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

def generate_parquet_samples(base_path, max_i=None, start_iter=2):
    """
    从指定路径下读取meta_asl/qa_data_i_train.parquet或qa_data_i.jsonl文件，
    优先尝试parquet文件，找不到则尝试jsonl文件，
    其中i是模3余2的整数（2, 5, 8, 11, ...），
    从每个文件中随机采样1000个数据，保存为question_iter_i.parquet
    
    Args:
        base_path (str): 基础路径
        max_i (int, optional): 最大i值，如果不指定则自动查找所有存在的文件
    """
    base_path = Path(base_path)
    meta_asl_path = base_path / "meta_asl"
    
    if not meta_asl_path.exists():
        print(f"错误: 路径 {meta_asl_path} 不存在")
        return
    
    # 如果没有指定max_i，自动查找所有符合条件的文件
    if max_i is None:
        max_i = 100  # 设置一个合理的上限
    
    processed_files = []
    
    # 生成所有模3余2的数字：2, 5, 8, 11, 14, ...
    for i in range(start_iter, max_i + 1):
        if i % 3 == 2:  # 模3余2的条件
            # 优先尝试parquet文件
            parquet_file = meta_asl_path / f"qa_data_{i}_train.parquet"
            jsonl_file = meta_asl_path / f"qa_data_{i}.jsonl"
            
            input_file = None
            if parquet_file.exists():
                input_file = parquet_file
                file_type = "parquet"
            elif jsonl_file.exists():
                input_file = jsonl_file
                file_type = "jsonl"
            
            if input_file is not None:
                try:
                    print(f"正在处理文件: {input_file} (格式: {file_type})")
                    
                    # 使用统一的读取函数
                    df = read_data_file(input_file)
                    print(f"文件 {input_file.name} 共有 {len(df)} 行数据")
                    
                    # INSERT_YOUR_CODE
                    # 过滤掉[reward_model][ground_truth]为"unknown"的样本
                    if "reward_model" in df.columns and "ground_truth" in df["reward_model"].iloc[0]:
                        df = df[df["reward_model"].apply(lambda x: x.get("ground_truth", None) != "unknown")]
                        print(f"过滤后剩余 {len(df)} 行数据（去除ground_truth为'unknown'的样本）")

                    # 随机采样200个（如果数据不足200个，则取全部）
                    sample_size = min(1000, len(df))
                    if sample_size < len(df):
                        sampled_df = df.sample(n=sample_size, random_state=37)
                        print(f"从 {len(df)} 行中随机采样了 {sample_size} 行")
                    else:
                        sampled_df = df
                        print(f"数据不足200行，使用全部 {len(df)} 行数据")
                    
                    # 修改reward_model字段中的style为"rule"
                    sampled_df = modify_reward_model_style(sampled_df)
                    
                    # 保存为新文件
                    output_file = meta_asl_path / f"question_iter_{i}.parquet"
                    sampled_df.to_parquet(output_file, index=False)
                    print(f"已保存到: {output_file}")
                    
                    processed_files.append((i, len(df), sample_size, str(output_file)))
                    
                except Exception as e:
                    print(f"处理文件 {input_file} 时出错: {e}")
            else:
                print(f"文件不存在: qa_data_{i}_train.parquet 和 qa_data_{i}.jsonl")
    
    # 打印处理结果摘要
    print("\n=== 处理结果摘要 ===")
    if processed_files:
        print(f"成功处理了 {len(processed_files)} 个文件:")
        for i, original_size, sample_size, output_path in processed_files:
                                    print(f"  qa_data_{i}: {original_size} -> {sample_size} 行 -> {Path(output_path).name}")
    else:
        print("没有找到符合条件的文件")

def main():
    parser = argparse.ArgumentParser(description="从parquet或jsonl文件中随机采样数据")
    parser.add_argument("--path", help="包含meta_asl目录的基础路径", default="../../exp/asl_grm/")
    parser.add_argument("--start_iter", help="输出路径", default=1)
    parser.add_argument("--max-i", type=int, default=100, 
                       help="最大i值 (默认自动查找所有文件)")
    
    args = parser.parse_args()
    
    generate_parquet_samples(args.path, args.max_i, args.start_iter)

if __name__ == "__main__":
    main()
