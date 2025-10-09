# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import json
import random

import requests


def mm_web_search(query, filter=3, xhs_size=5, web_size=5, snippet=False, mkt=""):
    """https://apifox.com/apidoc/shared-90b3decf-
    add1-44c8-8a88-d1582fe8692a/api-185854794.

    Args:
        query (_type_): _description_
        filter (int, optional): _description_. Defaults to 1.
        xhs_size (int, optional): _description_. Defaults to 5.
        web_size (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """

    url = "https://agi-redbot.devops.xiaohongshu.com/redbot/backend/api/search/hybrid_web_search"
    # url = "https://agi-redbot.devops.beta.xiaohongshu.com/redbot/backend/api/search/hybrid_web_search"
    json_payload = {
        "query": query,
        "xhs": {
            "size": xhs_size,
            "filters": filter,  # 1 图文，2 视频，3 视频+图片
            "threshold": 0.285,
        },
        "web": {
            "size": web_size,
            "snippet": snippet,
            "crawlTimeOut": 5000,
            "contentLength": -1,
            "mkt": "en-US",
        },
        "politic": None,
    }
    if mkt == "en":
        json_payload["web"]["mkt"] = "en-US"
    payload = json.dumps(json_payload)
    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    return result["data"]


def retrieve(query, topk=3, return_scores=True):
    import json

    import requests

    ips = ["22.0.198.95"]  # "22.1.88.166", #, "22.4.102.195"
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
    # obs = mm_web_search("初恋男友忽冷忽热 分分合合的感情还能持续多久", filter=3)
    # for line in obs:
    #     print(line["relevanceScore"])
    #     print(line["url"])

    # 单个查询
    results = retrieve("university in Beijing")

    # # 多个查询
    # results = retrieve(["什么是机器学习?", "深度学习的应用"], topk=5)

    # # 指定远程服务
    # results = retrieve("查询内容", host="192.168.1.100", port=8000)
