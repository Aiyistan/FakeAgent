import json
from typing import Dict, Any

from .base import BaseAgent
from ..utils.json_tools import parse_model_json_response


class FactChecker(BaseAgent):
    """事实准确性验证专家"""

    def __init__(self):
        super().__init__(
            role="事实准确性验证专家",
            goal="检查视频中陈述的事实是否准确，识别可能的虚假信息或误导性内容。",
            backstory="你是一位专业的事实核查专家，擅长验证信息的准确性和真实性。你能够通过分析视频内容来识别事实错误或故意传播的虚假信息。"
        )

    def analyze(self, inputs: Dict[str, Any]) -> str:
        video_title = inputs.get('video_title', '无')
        transcription = inputs.get('video_transcription', '无')
        frame_descriptions = inputs.get('frame_descriptions', [])

        # 构建分析任务prompt
        task_description = f"""
            检查视频中陈述的事实准确性，并判断是否需要外部证据：

            视频标题: {video_title}

            音频转录内容: {transcription}

            视频画面描述: {'; '.join(frame_descriptions) if frame_descriptions else '无画面描述'}

            请检查：
            1. 视频中陈述的事实是否逻辑错误
            2. 是否存在明显的知识型错误
            3. 是否有误导性或夸大的表述
            4. 是否存在不同事件的拼接
            5. 整体内容的真实性如何
            6. **重要**：判断是否可以基于当前信息明确确定视频真假
            7. 如果可以明确判断，给出明确的结论（真/假）
            8. 如果不能明确判断，说明原因并提取核心观点用于后续检索

            一定注意：
            你要清楚你自己的知识边界，对于超过自己知识范围之外或者视频时间在你掌握知识时间之外的内容不胡乱猜测判断,可借助外部证据进行事实核查。
        """

        expected_output = """
            请按照以下JSON格式输出，除了此JSON对象外不要返回任何其他内容。
            {
                "is_real": "是否为真（是/否/不确定）",
                "confidence_score": "判断结果自信度，百分比字符串，如85%",
                "fact_checking_results": ["事实检查结果1的描述", "事实检查结果2的描述", ...],
                "need_external_evidence": "是否需要外部证据进行判断（是/否）",
                "external_evidence_core_ideas": "需要外部证据的核心观点列表"
            }
        """


        prompt = task_description + "\n\n" + expected_output

        response = self.call_llm(prompt)

        response_json = parse_model_json_response(response)
        if response_json is None:
            return json.dumps({"error": "JSON解析失败", "raw_response": response}, indent=4, ensure_ascii=False)
        return json.dumps(response_json, indent=4, ensure_ascii=False)