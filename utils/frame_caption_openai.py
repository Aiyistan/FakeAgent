import base64
from pathlib import Path
from typing import List, Union
import openai


class VisionInferencer:
    """极简 OpenAI 兼容视觉推理客户端（仅支持 base64 图像输入）"""

    def __init__(self, api_base: str, api_key: str, model: str, timeout: int = 60):
        self.client = openai.OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """将图像文件直接编码为 base64（不检查格式/大小/颜色模式）"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def infer(self, image_path: Union[str, Path], prompt: str = "Describe this image.", **kwargs) -> str:
        """对单张图像进行推理"""
        b64 = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            **kwargs
        )
        return response.choices[0].message.content

    def batch_infer(self, image_paths: List[Union[str, Path]], prompt: str = "Describe this image.", **kwargs) -> List[str]:
        """批量推理（顺序执行）"""
        return [self.infer(path, prompt, **kwargs) for path in image_paths]