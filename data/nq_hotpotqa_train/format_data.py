import pandas as pd
import json
import random

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

THINK_SYS = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
## Formats
Your output should be a combination of the following formats:
1. <think>your reasoning thoughts</think>
2. <answer>YOUR ANSWER</answer>
## Tasks
Answer the user's question.
You can use <think></think> as many times as you want, but you must use <answer></answer> once and only once at the end of your response.
Your answer should be concise and to the point, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

# RETRIEVAL_SYS = """
# Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
# After reasoning, if you find you lack some knowledge, you can call a search engine by outputing the query in the following json format, for example:
# <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": [\"China capital\", \"China largest city\", ...],\n    }\n}\n</tool_call>
# It will then return the top searched results between <tool_response> and </tool_response>. You can search as many times as you want. 
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. 
# For example, <answer> Beijing </answer>.
# """

if __name__ == "__main__":

    for style in ["rule", "gen_rm"]:
        # 读取 parquet 文件
        for split in ["train", "test"]:
            
            df = pd.read_parquet(f'./{split}.parquet')
            examples = [row.to_dict() for _, row in df.iterrows()]
            # if split == "test":
            #     examples = random.sample(examples, 100)
            # 打印一个样本的结构
            print("数据样本示例：")
            #print(json.dumps(str(examples[0]), indent=2, ensure_ascii=False))

            # 修改 reward_model 下的 style 字段
            def modify_style(x):
                if isinstance(x, dict) and 'reward_model' in x:
                    x['reward_model']['style'] = style
                    x['data_source'] = f'asl-{x["data_source"]}'
                    x['golden_answers'] = x['golden_answers'].tolist()
                    x["env_name"] = "Retrieval"
                    x["prompt"] = {
                                "role": "system",
                                "content": RETRIEVAL_SYS
                                },{
                                    "role": "user",
                                    "content": x['question'],
                                },
                    x['reward_model']['ground_truth'] = x['reward_model']['ground_truth']["target"][0]
                return x

            # 应用修改
            examples = [modify_style(e) for e in examples]

            print("\n修改后的数据样本示例：")
            #json.dump(examples[:10], open(f"sample_{style}_{split}.json", "w"), indent=2, ensure_ascii=False)

            # 将修改后的examples转换回DataFrame
            df = pd.DataFrame(examples)

            # 保存处理后的数据
            output_path = f'./{split}_{style}.parquet'
            df.to_parquet(output_path)
            print(f"\n数据已保存到：{output_path}")
