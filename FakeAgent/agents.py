import os
import json
import re
import requests
from typing import Dict, List, Any
from openai import OpenAI
import dotenv

# 加载环境变量
dotenv.load_dotenv()

def parse_model_json_response(response_text, max_retries=3):
    """
    解析大模型输出的JSON响应
    
    Args:
        response_text (str): 模型返回的文本内容
        max_retries (int): 最大重试解析次数
    
    Returns:
        dict: 解析后的JSON字典，如果解析失败返回None
    
    Raises:
        ValueError: 当JSON格式严重错误时
    """
    if not response_text or not response_text.strip():
        return None
    
    text = response_text.split('</think>')[-1].strip()
    
    # 尝试直接解析
    try:
        parsed = json.loads(text)
        # 确保解析结果是字典或列表
        if isinstance(parsed, (dict, list)):
            return parsed
        else:
            return None
    except json.JSONDecodeError:
        pass  # 继续尝试其他方法
    
    # 常见情况1：JSON被代码块标记包围
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 常见情况2：只有JSON对象但没有代码块标记
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试修复常见的JSON格式问题
    for attempt in range(max_retries):
        try:
            # 移除可能的多余字符
            cleaned_text = re.sub(r'^[^{]*', '', text)  # 移除JSON前的文本
            cleaned_text = re.sub(r'[^}]*$', '', cleaned_text)  # 移除JSON后的文本
            cleaned_text = cleaned_text.strip()
            
            # 尝试解析清理后的文本
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"JSON解析失败: {e}")
                print(f"原始文本: {text}")
                return None
            continue
    
    return None

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

            # 执行工具函数
            function_response = self.tool_functions[function_name](**function_args)

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

        response = self.call_llm(prompt)
        response_json = parse_model_json_response(response)
        if response_json is None:
            return json.dumps({"error": "JSON解析失败", "raw_response": response}, indent=4, ensure_ascii=False)
        return json.dumps(response_json, indent=4, ensure_ascii=False)
