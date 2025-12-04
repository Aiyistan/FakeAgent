import os
import json
import requests
from typing import Dict, Any

from .base import BaseAgent


class Retriever(BaseAgent):
    """外部证据检索器"""

    def __init__(self):
        super().__init__(
            role="外部证据检索器",
            goal="根据关键词和核心观点，从网络搜索中检索相关外部证据。",
            backstory="你是一位专业的信息检索专家，擅长从各种数据源中快速找到相关证据。你能够根据给定的关键词和核心观点，高效地检索到支持或反驳视频内容的外部证据。"
        )

        # 从环境变量获取API keys，如果没有则使用默认key
        self.api_keys = os.getenv("SERPER_API_KEYS", "").split(",")
        self.model = os.getenv("MODEL")
        self.current_key_index = 0

        # 定义联网搜索工具
        self.tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "当需要获取实时信息、最新数据或网络内容时使用此工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要搜索的关键词或问题"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

        # 工具函数映射
        self.tool_functions = {
            "web_search": self.web_search
        }

    def web_search(self, query: str) -> str:
        """执行网络搜索并返回格式化结果，支持API key切换"""
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})

        # 尝试所有API keys
        for attempt in range(len(self.api_keys)):
            headers = {
                'X-API-KEY': self.api_keys[self.current_key_index],
                'Content-Type': 'application/json'
            }

            try:
                response = requests.post(url, headers=headers, data=payload)
                response.raise_for_status()
                results = response.json()

                # 解析并格式化结果
                formatted_results = []
                for item in results.get("organic", [])[:5]:  # 增加到5个结果
                    formatted_results.append(
                        f"标题: {item.get('title', '')}\n"
                        f"摘要: {item.get('snippet', '')}\n"
                        f"链接: {item.get('link', '')}"
                    )

                return "\n\n".join(formatted_results)

            except Exception as e:
                print(f"API key {self.current_key_index} 失败: {str(e)}")
                # 切换到下一个key
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        # 所有keys都失败了
        return f"所有API keys都失败，无法执行搜索: {str(e)}"

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_title = inputs.get('video_title', '无')
        # transcription = inputs.get('video_transcription', '无')
        # frame_descriptions = inputs.get('frame_descriptions', [])
        # keywords = inputs.get('keywords', '')
        core_ideas = inputs.get('core_ideas', '')

        # 构建检索任务prompt
        user_message = f"""
            根据视频的核心观点，检索相关外部证据：

            视频标题:
            {video_title}

            核心观点:
            {core_ideas}

            请执行以下步骤：
            1. 进行网络搜索
            2. 评估检索到的证据与视频内容的相关性
            3. 整理最有价值的外部证据

            请使用web_search工具来搜索相关信息，然后基于搜索结果给出分析报告。
        """

        return self.chat_with_tools(user_message)

    def chat_with_tools(self, user_message: str) -> str:
        """带工具调用的聊天函数"""
        messages = [{"role": "user", "content": user_message}]

        # 发送请求到LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 检查是否有工具调用
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用工具: {function_name}")
            # print(f"📝 参数: {function_args}")

            # 执行工具函数
            function_response = self.tool_functions[function_name](**function_args)
            # print(f"📊 搜索结果: {function_response[:200]}...")

            # 将工具结果添加到对话历史
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response
            })

            # 获取最终回复
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            return final_response.choices[0].message.content
        else:
            return response_message.content