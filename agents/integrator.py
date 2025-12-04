import json
from typing import Dict, Any

from .base import BaseAgent
from ..utils.json_tools import parse_model_json_response


class Integrator(BaseAgent):
    """分析结果整合专家"""

    def __init__(self):
        super().__init__(
            role="分析结果整合专家",
            goal="综合各个分析结果和外部证据，给出最终的视频真假判断。",
            backstory="你是一位专业的分析整合专家，擅长综合多个来源的信息并做出综合判断。你能够权衡不同分析结果的重要性，整合外部证据，并给出最终的权威结论。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        consistency = inputs.get('consistency_analysis', '')
        ai_detection = inputs.get('ai_detection', '')
        offensive = inputs.get('offensive_language_detection', '')
        fact_checking = inputs.get('fact_checking', '')
        external_evidence = inputs.get('external_evidence', '未检索外部证据')
        suspicious_segments = inputs.get('suspicious_segments', '未定位可疑片段')

        构建整合任务prompt
        task_description = f"""
            综合各个分析结果，给出最终判断：

            一致性分析结果: {consistency}

            AI检测结果: {ai_detection}

            冒犯性语言检测结果: {offensive}

            事实检查结果: {fact_checking}

            外部证据: {external_evidence}

            虚假内容定位: {suspicious_segments}

            请执行以下步骤：
            1. 综合所有分析结果
            2. 如果有外部证据，将其纳入综合判断
            3. 如果有虚假内容定位信息，将其纳入判断依据
            4. 给出最终的视频真假判断
            5. 提供详细的判断依据和置信度评估
            6. 生成完整的检测报告

            注意：由于对视频画面内容的分析是通过抽帧，所以可能出现信息遗漏，不要因此而影响最终结论，部分内容出现是正常，只有明确发现的可疑内容才会被纳入分析，未出现的内容不做猜测。
        """

        expected_output = """
            请按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "final_judgement": "视频是否为虚假内容的最终判断，不可以不确定（是/否）",
                "confidence_score": "总体置信度评估（0-100%）",
                "core_ideas": "视频核心观点内容",
                "analysis_summary": "详细判断依据总结",
                "external_evidence": "使用的外部证据（如果有）",
                "suspicious_segments": "可疑内容的时间区间（如果有）"
            }
        """

        prompt = task_description + "\n\n" + expected_output

        # print(prompt)

        response = self.call_llm(prompt)
        response_json = parse_model_json_response(response)
        if response_json is None:
            return json.dumps({"error": "JSON解析失败", "raw_response": response}, indent=4, ensure_ascii=False)
        return json.dumps(response_json, indent=4, ensure_ascii=False)