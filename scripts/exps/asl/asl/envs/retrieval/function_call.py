# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import json
import random

import requests

def retrieve(query, topk=3, return_scores=True):
    import json
    import requests

    ips = ["<your server ip>"]
    ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
    # 服务器地址
    url = f"http://{random.choice(ips)}:{random.choice(ports)}/retrieve"

    if isinstance(query, str):
        query = [query]
    # 请求数据
    data = {"queries": query, "topk": topk, "return_scores": return_scores}

    try:
        # 发送POST请求
        response = requests.post(url, json=data)

        # 检查响应
        result = json.loads(response.text)

        if result["error"]:
            raise Exception(result["result"])

    except Exception as e:
        print(f"******************Error occured in retrieval: {e}******************")
        print(query)
        print(response.text)
        result = {"result": [[{"document": {"id": "-1", "contents": "No relevant documents found"}, "score": 0.0}]]}

    return result["result"]


def function_call(functioncall):
    functioncall = set_default_functioncall_args(functioncall)
    observation = []
    if functioncall["name"] == "retrieve":
        function_results = retrieve(query=functioncall["arguments"]["query"], topk=functioncall["arguments"]["size"])
        flatten_results = []
        for i in range(len(function_results)):
            flatten_results.extend(function_results[i])
        all_results = []
        for cand_result in flatten_results:
            for result in all_results:
                if result["document"]["id"] == cand_result["document"]["id"]:
                    result["score"] += cand_result["score"]
                    break
            else:
                all_results.append(cand_result)
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[: functioncall["arguments"]["size"]]
        observation = all_results
    return observation


def set_default_functioncall_args(functioncall):
    if functioncall["name"] in ["retrieve"]:
        if "size" not in functioncall:
            functioncall["arguments"]["size"] = 3
        if "snippet" not in functioncall:
            functioncall["arguments"]["snippet"] = False
        if "filters" not in functioncall:
            functioncall["arguments"]["filters"] = 3
        if "mkt" not in functioncall:
            functioncall["arguments"]["mkt"] = ""
    return functioncall


if __name__ == "__main__":

    # 单个查询
    results = retrieve("university in Beijing")

    # # 多个查询
    # results = retrieve(["什么是机器学习?", "深度学习的应用"], topk=5)

