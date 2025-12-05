import json
from typing import Dict, Any

from agents.base import BaseAgent
from utils.json_tools import parse_model_json_response


class ConsistencyAnalyzer(BaseAgent):
    """一致性分析专家"""

    def __init__(self):
        super().__init__(
            role="视频一致性分析专家",
            goal="分析视频画面、音频和标题之间的一致性，识别是否存在不匹配或矛盾之处。",
            backstory="你是一位专业的多模态内容分析专家，擅长比较视频的不同组成部分（画面、音频、标题）。你能够敏锐地发现内容之间的不一致性，这往往是虚假视频的重要特征。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_title = inputs.get('video_title', '无')
        transcription = inputs.get('video_transcription', '无')
        frame_descriptions = inputs.get('frame_descriptions', [])

        # 构建分析任务prompt
        task_description = f"""
            分析视频画面、音频和标题之间的一致性：

            视频标题: {video_title}

            音频转录内容: {transcription}

            视频画面描述: {'; '.join(frame_descriptions) if frame_descriptions else '无画面描述'}

            请分析：
            1. 画面、音频和标题之间的是否一致
            2. 发现的不一致之处列表
            3. 给出一致性评分（0-100%）
        """

        expected_output = """
            按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "is_consistent": "是否一致（是/否）",
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