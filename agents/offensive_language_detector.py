import json
from typing import Dict, Any

from agents.base import BaseAgent
from utils.json_tools import parse_model_json_response


class OffensiveLanguageDetector(BaseAgent):
    """冒犯性语言识别专家"""

    def __init__(self):
        super().__init__(
            role="冒犯性语言识别专家",
            goal="识别视频中可能的冒犯性、歧视性或不当语言内容。",
            backstory="你是一位内容审核专家，擅长识别各种形式的冒犯性语言和不当内容。你对语言敏感性有深入理解，能够准确识别潜在的有害内容。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_title = inputs.get('video_title', '无')
        transcription = inputs.get('video_transcription', '无')
        frame_descriptions = inputs.get('frame_descriptions', [])

        # 构建分析任务prompt
        task_description = f"""
            识别视频中可能的冒犯性语言：

            视频标题: {video_title}

            音频转录内容: {transcription}

            视频画面描述: {'; '.join(frame_descriptions) if frame_descriptions else '无画面描述'}

            请识别：
            1. 是否存在歧视性语言
            2. 是否包含冒犯性词汇或表达
            3. 是否有不当的内容描述
            4. 整体内容是否符合社区标准
        """

        expected_output = """
            按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "is_offensive": "是否存在冒犯性语言（是/否）",
                "confidence_score": "判断结果自信度，百分比字符串，如85%",
                "detection_results": ["检测结果1的描述", "检测结果2的描述", ...]
            }
        """

        prompt = task_description + "\n\n" + expected_output

        response =  self.call_llm(prompt)
        response_json = parse_model_json_response(response)
        if response_json is None:
            return json.dumps({"error": "JSON解析失败", "raw_response": response}, indent=4, ensure_ascii=False)
        return json.dumps(response_json, indent=4, ensure_ascii=False)