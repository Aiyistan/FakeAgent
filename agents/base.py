import os
from typing import Dict, Any
from openai import OpenAI
import dotenv

# 加载环境变量
dotenv.load_dotenv()


class BaseAgent:
    """基础代理类"""

    def __init__(self, role: str, goal: str, backstory: str):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.model = os.getenv("MODEL", "Qwen3-8B")

    def analyze(self, inputs: Dict[str, Any]) -> str:
        """执行分析，返回报告"""
        raise NotImplementedError("子类必须实现analyze方法")

    def call_llm(self, prompt: str) -> str:
        """调用大模型"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": f"你是{self.role}。\n\n{self.backstory}\n\n你的目标是：{self.goal}.请根据提供的抽取的信息进行判断，不要过于悲观，不要对于各种信息的判断过于谨慎不确定。如果不能找到证据就认为是真的；没提供的时间、位置等信息可能未抽取到，不代表虚假。不要直接认为掌握的知识和掌握信息时间之外为假。"},
                    {"role": "system", "content": f"你是{self.role}。\n\n{self.backstory}\n\n你的目标是：{self.goal}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return f"分析失败：{str(e)}"