# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import json

from src.verl.agent.tool_envs import ToolBase, extract_tool_call_contents

from .function_call import function_call


class Retrieval(ToolBase):
    name = "Retrieval"
    action_start = "<tool_call>"
    action_end = "</tool_call>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    doc_start = "<tool_response>"
    doc_end = "</tool_response>"

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return "", 0.0, True, {}

        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_list:
            # print(f' [DEBUG] no action_list in {action_string=}')
            return "", 0.0, True, {}

        # Create a new list for parsed actions instead of modifying action_list while iterating
        parsed_actions = []
        for action in action_list:
            try:
                parsed_action = json.loads(action.strip())
                parsed_actions.append(parsed_action)  # Add to new list instead
            except json.JSONDecodeError:
                pass

        search_results = []
        # Use the parsed_actions list in this loop instead of action_list
        for action in parsed_actions:
            result = []
            try:
                result = function_call(action)
            except Exception:
                pass
            search_results.append(result)

        docs_messages = []
        for search_result in search_results:
            observation_messages = self._observation2messages(search_result)
            docs_messages.append(observation_messages)

        if len(docs_messages) == 0:
            docs_messages = [{"role": "tool", "content": "No search results found"}]

        return docs_messages, 0.0, False, search_results

    def _observation2messages(self, search_result):
        # search_result: list of observation
        obs_strings = ""
        for obs in search_result:
            obs_string = f"\nid: {obs['document']['id']}\ncontent:\n{obs['document']['contents']}\n"
            obs_strings += obs_string

        # obs_strings = f"Here are the search results:\n{obs_strings}\nPlease use <think></think> to summarize the current situation and then make the next action."

        return {"role": "tool", "content": obs_strings}

    def reset(self, *args, **kwargs):
        pass


def main():
    #     test_action_string_bing = """
    # <tool_call>
    # {
    #     "name": "search_bing",
    #     "arguments": {
    #         "query": ["最新的人工智能研究进展", "LLM发展"],
    #         "snippet": "False"
    #     }
    # }
    # </tool_call>
    # <tool_call>
    # {
    #     "name": "search_xhs",
    #     "arguments": {
    #         "query": ["AGI发展"],
    #         "size": 3
    #     }
    # }
    # </tool_call>
    # """

    test_action_string_bing = """
<tool_call>
[
  {
    "name": "search_xhs",
    "arguments": {
      "query": ["中岛美雪 生命的别名 丝 发行年份"]
    }
  },
  {
    "name": "search_bing",
    "arguments": {
      "query": ["中岛美雪 生命的别名 丝 发行年份"]
    }
  }
]
</tool_call>
    """
    print("\n\n" + "=" * 50)
    print("测试场景3: 搜索Bing")
    print("=" * 50)
    print(f"输入: {test_action_string_bing}")

    retrieval_tool = Retrieval("retrieval", "检索工具", {})

    docs_string, reward, done, search_results = retrieval_tool.execute(test_action_string_bing)

    print("\n输出:")
    print(f"docs_string: {docs_string}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"search_results长度: {len(search_results)}")


if __name__ == "__main__":
    main()
