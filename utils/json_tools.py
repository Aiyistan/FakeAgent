import json
import re


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