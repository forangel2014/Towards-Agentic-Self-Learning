# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

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



# RETRIEVAL_SYS = """
# Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
# After reasoning, if you find you lack some knowledge, you can call a search engine by outputing the query in the following json format, for example:
# <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": [\"China capital\", \"China largest city\", ...],\n    }\n}\n</tool_call>
# It will then return the top searched results between <tool_response> and </tool_response>. You can search as many times as you want. 
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. 
# For example, <answer> Beijing </answer>.
# """

#Your output should be a combination of the following formats:


THINK_SYS = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 
## Formats
Your output should be a combination of the following formats:
1. <think>\nyour reasoning thoughts\n</think>
3. <answer>\nYOUR ANSWER\n</answer>
You can use <think> as many times as you want, but you must use <answer> once and only once at the end of your response.
Your answer should be concise and to the point, without detailed illustrations. For example, <answer>\nBeijing\n</answer>.
"""

# verify_prompt_template = """
# **Please carefully evaluate the correctness of the provided answer to the question:**

# Question:
# {question_prompt}

# Answer:
# {completion}

# **First output your analysis and reasoning process, and then draw your conclusion in the following format:**
# <answer>
# conclusion: correct/wrong
# </answer>
# """

verify_prompt_template = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me verify the correctness of my student's answer. 
Use the tool and thinking to analyze, then draw your conclusion in the following format:**
<answer>
conclusion: correct/wrong
</answer>
"""

PROMPT_GENERATOR_SYS = """
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
## Tasks
generate a question and answer that satisfying the user's request.
You can use <think></think> and <tool_call></tool_call> as many times as you want, but you must use <question></question> and <answer></answer> once and only once at the end of your response.
Your question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Your answer should be concise and to the point, without detailed illustrations. 
For example:
<question>What is the capital of China?</question>
<answer>Beijing</answer>
## Formats
Your output should be a combination of the following formats:
1. <think>your reasoning thoughts</think>
2. <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": \"Beijing cuisine\"\n    }\n}\n</tool_call>
3. <question>YOUR GENERATED QUESTION</question>
4. <answer>THE ANSWER TO THE GENERATED QUESTION</answer>
"""

import openai
client = openai.OpenAI(api_key="<your key>", base_url="<your url>")


if __name__ == '__main__':


#     meta_question = """
# Please generate a question about {topic}. 
# The question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
# Use the tool to help you generate the question, ensure that the question is solvable and the tool is necessary to answer the question.
# Output with the following format:
# <answer>
# ...a generated question...
# </answer>
# """

    meta_question = """
Please generate a question about {topic}. 
The question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Use the tool to help you generate the question, ensure that the question is solvable and the tool is necessary to answer the question.
Output with the following format:
<question>
...a generated question...
</question>
<answer>
...the answer to the generated question...
</answer>
"""

    data = [{
        "data_source": "asl",#data_source,
        "prompt": [{
            "role": "system",
            "content": PROMPT_GENERATOR_SYS#RETRIEVAL_SYS
        },{
            "role": "user",
            "content": meta_question.format(topic=topic),
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "meta",
            "ground_truth": "unknown"
        },
        "extra_info": {
            'split': "train",
            'index': 0,
        },
        "env_name": "Retrieval"
    } for topic in ["China", "Britain", "France", "Germany", "Italy", "Japan", "a Banker", "a Lawyer", "a Doctor", "a Teacher", 
                    "an Engineer", "a Scientist", "an Artist", "a Musician", "a Writer", "a Director", "an Actor", "an Actress", "a Producer", 
                    "a Composer", "a Lyricist", "a Dancer", "a Singer", "a Painter", "a Sculptor", "a Photographer", "a Videographer", 
                    "an Editor", "a Journalist", "a Reporter", "a Commentator", "an Analyst", "a Consultant", "a Strategist", "an Architect", 
                    "a Designer", "a Builder", "a Carpenter", "an Electrician", "a Trader", 
                    "a Merchant", "a Shopkeeper", "Korea", "Russia", "Spain", "United States", "India", "Brazil", "a Planet", "a Plant",
                    "an Animal", "a Food", "a type of Weather", "an Art creation", "a Music song", "a Movie", 
                    "a TV series", "a Game", "a Health status", "a Sports event", 
                    "a Film", "Turkey", "Greece", "Egypt", "Iran", "Iraq", "Israel", "Palestine", "Jordan", "Lebanon", "Syria", "Turkey", "Greece"]]

    meta_dataset = datasets.Dataset.from_list(data)

    meta_only_dataset = datasets.Dataset.from_list(data*10)
    meta_only_dataset.to_parquet(os.path.join('./', 'train.parquet'))
    meta_only_dataset.select(range(100)).to_parquet(os.path.join('./', 'test.parquet'))

    # meta_only_dataset = datasets.Dataset.from_list(data*3)
    # meta_only_dataset.to_parquet(os.path.join('./', 'asl_online_train.parquet'))
    # meta_only_dataset.select(range(5)).to_parquet(os.path.join('./', 'asl_online_test.parquet'))

