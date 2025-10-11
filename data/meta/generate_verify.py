
import re
import os
import datasets
import random
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

RETRIEVAL_SYS = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
## Tools
You are provided with function signatures within <tools></tools> tags:
<tools>
Name:retrieve
Description: Retrieve relevant information from the locally deployed knowledge base based on the provided list of search terms.
Input:{'query': {'Optional/Required': 'required', 'Parameter Description': 'search terms', 'Parameter Type': 'str'}}
</tools>
For each function call, you should call and then include the json format inputs within <tool_call></tool_call> tags, for example:
<tool_call>{\n  \"name\": tool['name'],\n  \"arguments\": tool['arguments']\n}</tool_call>
For each function call, the result will be returned in the <tool_response></tool_response> tags. 
## Formats
Your output should be a combination of the following formats:
1. <think>your reasoning thoughts</think>
2. <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": \"Beijing cuisine\"\n    }\n}\n</tool_call>
3. <answer>YOUR ANSWER</answer>
## Tasks
Answer the user's question.
You can use <think></think> and <tool_call></tool_call> as many times as you want, but you must use <answer></answer> once and only once at the end of your response.
Your answer should be concise and to the point, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

verify_prompt_template = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me verify the correctness of my student's answer. 
Use the tool and thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: correct/wrong
</answer>
"""

import openai
client = openai.OpenAI(api_key="<your key>", base_url="<your url>")


def make_map_fn_verify_correct(split):

    def process_fn(example, idx):
        example['question'] = example['question'].strip()
        if example['question'][-1] != '?':
            example['question'] += '?'
        question = example['question']
        solution = {
            "target": example['golden_answers'],
        }

        data = {
            "data_source": "asl",#data_source,
            "prompt": [{
                "role": "system",
                "content": RETRIEVAL_SYS
            },{
                "role": "user",
                "content": verify_prompt_template.format(question_prompt=question, completion=example['golden_answers'][0]),
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": "conclusion: correct"
            },
            "extra_info": {
                'split': split,
                'index': idx,
            },
            "env_name": "Retrieval"
        }
        return data

    return process_fn

def make_map_fn_verify_wrong(split):

    def process_fn(example, idx):
        example['question'] = example['question'].strip()
        if example['question'][-1] != '?':
            example['question'] += '?'
        question = example['question']
        solution = {
            "target": example['golden_answers'],
        }
        
        wrong_answer = None
        while wrong_answer is None:
            try:
                # condition = random.choice(["nonsense words, i.e., ebruuyasdb<func>yyy", "\"No answer found.\"", "a nearly correct answer but still wrong answer", "a factual error answer"])
                condition = random.choice(["a nearly correct answer but still wrong answer", "a factual error answer"])
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Your are given a question and an answer. Please write a wrong answer to the question. The wrong answer should be: {condition}".format(condition=condition)},
                        {"role": "user", "content": f"Question: {question}\nAnswer: {example['golden_answers'][0]}\nWrong Answer:"}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                wrong_answer = response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")

        if random.random() < 0.01:
            print(f"Question: {question}\nAnswer: {example['golden_answers'][0]}\nWrong Answer: {wrong_answer}")

        data = {
            "data_source": "asl",#data_source,
            "prompt": [{
                "role": "system",
                "content": RETRIEVAL_SYS
            },{
                "role": "user",
                "content": verify_prompt_template.format(question_prompt=question, completion=wrong_answer),
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": "conclusion: wrong"
            },
            "extra_info": {
                'split': split,
                'index': idx,
            },
            "env_name": "Retrieval"
        }
        return data

    return process_fn

train_dataset = datasets.load_dataset("parquet", data_files=f'../nq_hotpotqa_train/train.parquet')["train"]

ratio = 0.01

# 从correct和wrong数据集中随机挑选ratio比例的数据
correct_size = len(train_dataset)
wrong_size = len(train_dataset)

# 计算要选择的数据量
selected_correct_size = int(correct_size * ratio)
selected_wrong_size = int(wrong_size * ratio)

print(f"size: {selected_correct_size}")

random.seed(37)  # 设置随机种子以保证可重复性

# 随机选择数据
selected_correct_indices = random.sample(range(correct_size), selected_correct_size)
selected_wrong_indices = random.sample(range(wrong_size), selected_wrong_size)

selected_verify_correct = train_dataset.select(selected_correct_indices)
selected_verify_wrong = train_dataset.select(selected_wrong_indices)

train_dataset_verify_correct = selected_verify_correct.map(function=make_map_fn_verify_correct('train'), with_indices=True)
train_dataset_verify_wrong = selected_verify_wrong.map(function=make_map_fn_verify_wrong('train'), with_indices=True, num_proc=100)

verify_dataset = datasets.concatenate_datasets([train_dataset_verify_correct, train_dataset_verify_wrong])

# 切分数据集：90%作为训练集，10%作为测试集
total_size = len(verify_dataset)
train_size = int(total_size * 0.9)

# 使用固定随机种子确保可重复性
random.seed(42)
indices = list(range(total_size))
random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

verify_train = verify_dataset.select(train_indices)
verify_test = verify_dataset.select(test_indices)

# 保存训练集和测试集
verify_train.to_parquet(os.path.join('./', 'verify_train.parquet'))
verify_test.to_parquet(os.path.join('./', 'verify_test.parquet'))

print(f"数据集切分完成:")
print(f"总数据量: {total_size}")
print(f"训练集: {len(verify_train)} ({len(verify_train)/total_size*100:.1f}%)")
print(f"测试集: {len(verify_test)} ({len(verify_test)/total_size*100:.1f}%)")

# 保持原有的完整数据集文件
verify_dataset.to_parquet(os.path.join('./', 'verify.parquet'))

print(f"correct_cnt: {correct_size}, wrong_cnt: {wrong_size}")
all_train_dataset = datasets.concatenate_datasets([train_dataset_verify_correct, train_dataset_verify_wrong])
# 随机保存10个case为json格式，用于人工观察

# 随机选择10个索引
total_samples = len(all_train_dataset)
sample_indices = random.sample(range(total_samples), min(500, total_samples))

# 提取这10个样本
sample_data = [all_train_dataset[i] for i in sample_indices]

# 保存为JSON文件
with open(os.path.join('./', 'sample_10_cases.json'), 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(sample_data)} 个样本到 sample_10_cases.json 文件中")