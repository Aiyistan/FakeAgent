import json
from typing import Dict, Any

from agents.base import BaseAgent
from utils.json_tools import parse_model_json_response


class Locator(BaseAgent):
    """虚假内容定位器"""

    def __init__(self):
        super().__init__(
            role="虚假内容定位器",
            goal="定位视频中虚假内容的具体位置",
            backstory="你是一位专业的视频分析专家，擅长精确定位视频中的问题内容。你能够结合分析结果和外部证据，准确定位视频中虚假内容的具体位置。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_content = inputs.get('video_content', '')
        analysis_results = inputs.get('analysis_results', '')
        external_evidence = inputs.get('external_evidence', '')

        # 构建定位任务prompt
        task_description = f"""
            根据分析结果和外部证据，定位视频中虚假内容的具体位置：

            视频内容:
            {video_content}

            分析结果:
            {analysis_results}

            外部证据:
            {external_evidence}

            请执行以下步骤：
            1. 结合分析结果和外部证据识别可疑内容
            2. 精确定位可疑内容的具体位置
        """

        expected_output = """
            请按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "suspicious_segments": ["可疑片段1的描述", "可疑片段2的描述", ...],
                "suspicious_timestamps": ["可疑片段1的起止时间戳", "可疑片段2的起止时间戳", ...],
            }
        """
        prompt = task_description + "\n\n" + expected_output

        response = self.call_llm(prompt)

        response_json = parse_model_json_response(response)
        if response_json is None:
            return json.dumps({"error": "JSON解析失败", "raw_response": response}, indent=4, ensure_ascii=False)
        return json.dumps(response_json, indent=4, ensure_ascii=False)