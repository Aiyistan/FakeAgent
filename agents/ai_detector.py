import json
from typing import Dict, Any

from .base import BaseAgent
from ..utils.json_tools import parse_model_json_response


class AIDetector(BaseAgent):
    """AI生成内容检测专家"""

    def __init__(self):
        super().__init__(
            role="AI生成内容检测专家",
            goal="检测视频中是否存在AI生成内容的迹象，包括深度伪造、AI合成等技术特征。",
            backstory="你是一位AI技术专家，熟悉各种AI生成技术的特征和痕迹。你能够通过分析视频的视觉和音频特征来识别AI生成的内容。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_title = inputs.get('video_title', '无')
        transcription = inputs.get('video_transcription', '无')
        frame_descriptions = inputs.get('frame_descriptions', [])

        # 构建分析任务prompt
        task_description = f"""
            检测视频中是否存在AI生成内容的迹象：

            视频标题: {video_title}

            音频转录内容: {transcription}

            视频画面描述: {'; '.join(frame_descriptions) if frame_descriptions else '无画面描述'}

            请检测：
            1. 画面中是否存在AI生成的特征（如异常的面部表情、不自然的动作）
            2. 音频中是否存在AI合成的迹象
            3. 整体内容是否显示出AI生成的模式
        """

        expected_output = """
            按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "is_ai_generated": "是否存在AI生成内容（是/否）",
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