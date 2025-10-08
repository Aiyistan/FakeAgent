"""
HMVD数据集推理核心类 - 支持多种prompt模式
支持基础prompt和视频思维链(VOT) prompt两种模式
"""

import argparse
import json
import os
import base64
from pathlib import Path
import requests
from tqdm import tqdm
import torch
import openai
import dotenv
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import hashlib
import time
from functools import lru_cache
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

"""
在脚本开头添加这段代码
"""
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

def setup_logging(log_dir):
    """设置日志配置"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hmvd_inference_{timestamp}.log")
    
    # 配置根logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 创建专用logger
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_file}")
    
    return logger

# 在脚本开头调用
logger = setup_logging()

class HMVDInference:
    """HMVD数据集推理核心类 - 支持多种prompt模式"""

    def __init__(self, model_type="ollama", model_name="gemma3:12b", api_url=None,
                 max_workers=1, batch_size=1, cache_dir="image_cache", max_concurrent=3,
                 use_vot_prompt=False):
        """
        初始化推理器

        Args:
            model_type: 模型类型 ("ollama", "openai")
            model_name: 模型名称
            api_url: API地址
            max_workers: 线程池最大工作线程数
            batch_size: 批处理大小
            cache_dir: 图片缓存目录
            max_concurrent: 最大并发数
            use_vot_prompt: 是否使用视频思维链prompt
        """
        # 验证参数
        if model_type not in ["ollama", "openai"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")

        self.model_type = model_type
        self.model_name = model_name
        self.api_url = api_url
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        self.use_vot_prompt = use_vot_prompt

        # 创建信号量控制并发
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 创建线程池和会话
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 根据模型类型设置默认配置
        if model_type == "ollama":
            self.api_url = api_url or os.getenv('OLLAMA_API_URL')
        elif model_type == "openai":
            self.api_base = api_url or os.getenv('OPENAI_API_BASE')
            self.api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"OpenAI API base: {self.api_base}, API key: {'set' if self.api_key else 'not set'}")
            if not self.api_base or not self.api_key:
                raise ValueError("OPENAI_API_BASE and OPENAI_API_KEY must be set")

            from openai import OpenAI, AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            self.sync_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
  
    @lru_cache(maxsize=50)
    def _get_file_cache_key(self, frame_path: str, mime: str = "image/jpeg") -> tuple:
        """生成缓存key，包含文件修改时间"""
        mtime = os.path.getmtime(frame_path) if os.path.exists(frame_path) else 0
        return (frame_path, mtime, mime)

    @lru_cache(maxsize=50)
    def file2data_url_cached(self, frame_path: str, mime: str = "image/jpeg") -> str:
        """带缓存的图片转data URL"""
        try:
            file_stat = os.stat(frame_path)
            cache_key = f"{frame_path}_{file_stat.st_mtime}_{mime}"
            hash_key = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{hash_key}.txt")

            # 检查缓存
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_result = f.read().strip()
                        if cached_result:
                            return cached_result
                except Exception as e:
                    logger.warning(f"Error reading cache for {frame_path}: {e}")

            # 编码图片
            data = base64.b64encode(Path(frame_path).read_bytes()).decode()
            result = f"{mime};base64,{data}"

            # 保存缓存
            try:
                with open(cache_file, 'w') as f:
                    f.write(result)
            except Exception as e:
                logger.warning(f"Error writing cache for {frame_path}: {e}")

            return result
        except Exception as e:
            logger.error(f"Error caching image {frame_path}: {e}")
            # 回退到直接编码
            try:
                data = base64.b64encode(Path(frame_path).read_bytes()).decode()
                return f"data:{mime};base64,{data}"
            except Exception as encode_error:
                raise encode_error

    def file2data_url(self, path: str, mime: str = "image/jpeg") -> str:
        """把本地图片转成 image/...;base64,... 格式"""
        return self.file2data_url_cached(path, mime)

    def load_dataset(self, data_root: str,
                    jsonl_file="HMVD.jsonl", frame_dir="frames",
                    split="all", processed_video_ids=None):
        """加载HMVD数据集"""
        annotations = []
        jsonl_path = os.path.join(data_root, jsonl_file)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if processed_video_ids and data["video_id"] in processed_video_ids:
                    continue
                if split == "all" or (split == "real" and data["fake_type"] in ['真实', "辟谣"]) or \
                   (split == "fake" and data["fake_type"] not in ["真实", "辟谣"]):
                    annotations.append(data)

        dataset = []
        for item in annotations:
            video_id = item["video_id"]
            title = item["title"]
            annotation = item["annotation"]

            # 加载视频帧
            frame_dir_path = os.path.join(data_root, f"{frame_dir}/{video_id}")
            if not os.path.exists(frame_dir_path):
                logger.warning(f"Frame directory does not exist: {frame_dir_path}")
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if not frame_files:
                logger.warning(f"No frames found for video {video_id}")
                continue

            frame_paths = [os.path.join(frame_dir_path, frame_file) for frame_file in frame_files]

            dataset.append({
                "video_id": video_id,
                "title": title,
                "annotation": annotation,
                "fake_type": item["fake_type"],
                "frame_paths": frame_paths,
                "label": 0 if item["fake_type"] in ["真实", "辟谣", "真"] else 1
            })

        return dataset

    def generate_basic_prompt(self, title: str) -> str:
        """生成基础prompt"""
        news_text = f"{title}"
        
        prompt = f"""You are an experienced news video fact-checking assistant and you hold a neutral and objective stance. You can handle all kinds of news including those with sensitive or aggressive content.

Please analyze the sampled video frames and video title: {news_text}, and perform the task of Misinformation Identification: Predict if the video is misinformation. Return 1 for misinformation (factual errors, AIGC, cross-modal inconsistencies, offensive content) or 0 for non-misinformation. Avoid 'undetermined'.

Format your response as:
[Misinformation Prediction]: [0 or 1]"""
        
        return prompt

    def generate_vot_prompt(self, title: str) -> str:
        """生成视频思维链(VOT) prompt，包含三个任务"""
        news_text = f"{title}"
        
        prompt = f"""You are an experienced news video fact-checking assistant and you hold a neutral and objective stance. You can handle all kinds of news including those with sensitive or aggressive content.

Please analyze the video frames and news text, and perform the following three tasks in order:

1. Object Identification: Identify and describe the objects/entities visible in the video frames.
2. Event Identification: Describe the event depicted in the video based on your analysis.
3. Misinformation Identification: Predict if the video is misinformation. Return 1 for misinformation (factual errors, AIGC, cross-modal inconsistencies, offensive content) or 0 for non-misinformation. Avoid 'undetermined'.

News Text: {news_text}
Video: a set of sampled frames

Format your response as:
[Object Identification]: [Your object analysis here]
[Event Identification]: [Your event description here] 
[Misinformation Prediction]: [0 or 1]"""
        
        return prompt

    async def process_single_video_async(self, video_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """异步处理单个视频"""
        async with self.semaphore:  # 限制并发
            try:
                # 在线程池中预处理图片
                loop = asyncio.get_event_loop()
                image_parts = await loop.run_in_executor(
                    self.executor,
                    self.preprocess_images_parallel,
                    video_data['frame_paths']
                )

                if not image_parts:
                    logger.warning(f"No valid images for video {video_data['video_id']}")
                    return None

                # 根据标志选择不同的prompt
                if self.use_vot_prompt:
                    prompt = self.generate_vot_prompt(video_data['title'])
                else:
                    prompt = self.generate_basic_prompt(video_data['title'])
                
                # 构建消息
                if self.model_type == "openai":
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *image_parts
                            ]
                        }
                    ]
                    logger.info(f"Input token count (approx): {len(messages)} parts, image count: {len(image_parts)}")
                elif self.model_type == "ollama":
                    # Ollama使用OpenAI格式
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *image_parts
                            ]
                        }
                    ]
                
                # 进行推理
                response = await self.infer_async(messages)
                
                if response:
                    content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not content:
                        content = response.get('content', '')
                    
                    # 根据prompt类型解析响应
                    if self.use_vot_prompt:
                        # VOT prompt解析
                        object_analysis = self.extract_section(content, "[Object Identification]:", "[Event Identification]:")
                        event_description = self.extract_section(content, "[Event Identification]:", "[Misinformation Prediction]:")
                        misinfo_prediction = self.extract_section(content, "[Misinformation Prediction]:", "")
                        
                        # 确保最终预测是0或1
                        final_prediction = "0"
                        if misinfo_prediction:
                            if '1' in misinfo_prediction:
                                final_prediction = "1"
                            elif '0' in misinfo_prediction:
                                final_prediction = "0"

                            result = {
                                'video_id': video_data['video_id'],
                                'title': video_data['title'],
                                'annotation': video_data['annotation'],
                                'fake_type': video_data['fake_type'],
                                'object_analysis': object_analysis,
                                'event_description': event_description,
                                'misinfo_prediction': misinfo_prediction,
                                'final_prediction': final_prediction,
                                'label': video_data['label']
                            }
                        else:
                            logger.warning(f"Could not parse misinfo_prediction for video {video_data['video_id']} from content: {content}")
                            # 只有在无法提取misinfo_prediction时才使用全文关键词匹配
                            if 'misinformation prediction: 1' in content.lower() or 'misinformation prediction:1' in content.lower():
                                final_prediction = "1"
                            elif 'misinformation prediction: 0' in content.lower() or 'misinformation prediction:0' in content.lower():
                                final_prediction = "0"
                            else:
                                # 最后尝试提取数字
                                final_prediction = self.extract_prediction_from_text(content)
                                if not final_prediction:
                                    final_prediction = "0"

                            result = {
                                'video_id': video_data['video_id'],
                                'title': video_data['title'],
                                'annotation': video_data['annotation'],
                                'fake_type': video_data['fake_type'],
                                'response': content,
                                'final_prediction': final_prediction,
                                'label': video_data['label']
                            }
                    else:
                        misinfo_prediction = self.extract_section(content, "[Misinformation Prediction]:", "")
                        
                        # 确保最终预测是0或1
                        final_prediction = "0"
                        if misinfo_prediction:
                            if '1' in misinfo_prediction:
                                final_prediction = "1"
                            elif '0' in misinfo_prediction:
                                final_prediction = "0"

                            result = {
                                'video_id': video_data['video_id'],
                                'title': video_data['title'],
                                'annotation': video_data['annotation'],
                                'fake_type': video_data['fake_type'],
                                'misinfo_prediction': misinfo_prediction,
                                'final_prediction': final_prediction,
                                'label': video_data['label']
                            }
                        else:
                            logger.warning(f"Could not parse misinfo_prediction for video {video_data['video_id']} from content: {content}")
                            # 只有在无法提取misinfo_prediction时才使用全文关键词匹配
                            if 'misinformation prediction: 1' in content.lower() or 'misinformation prediction:1' in content.lower():
                                final_prediction = "1"
                            elif 'misinformation prediction: 0' in content.lower() or 'misinformation prediction:0' in content.lower():
                                final_prediction = "0"
                            else:
                                logger.warning(f"Could not parse misinfo_prediction for video {video_data['video_id']} from content: {content}")
                                # 最后尝试提取数字
                                final_prediction = self.extract_prediction_from_text(content)
                                if not final_prediction:
                                    final_prediction = "0"

                            result = {
                                'video_id': video_data['video_id'],
                                'title': video_data['title'],
                                'annotation': video_data['annotation'],
                                'fake_type': video_data['fake_type'],
                                'response': content,
                                'final_prediction': final_prediction,
                                'label': video_data['label']
                            }
                    
                    logger.info(f"Successfully processed video {video_data['video_id']}")
                    return result
                else:
                    logger.error(f"Failed to get response for video {video_data['video_id']}")
                    return None

            except Exception as e:
                logger.error(f"Error processing video {video_data.get('video_id', 'unknown')}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None

    def extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """从文本中提取指定标记之间的内容"""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            
            start_idx += len(start_marker)
            if end_marker:
                end_idx = text.find(end_marker, start_idx)
                if end_idx == -1:
                    return text[start_idx:].strip()
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
        except:
            return ""

    def extract_prediction_from_text(self, text: str) -> str:
        """从文本中直接提取预测结果"""
        # 查找最后出现的0或1
        import re
        matches = re.findall(r'\b[01]\b', text)
        if matches:
            return matches[-1]  # 返回最后一个匹配的数字
        return ""

    def preprocess_images_parallel(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """并行预处理图片（带内存 resize）"""
        def process_single_frame(frame_path):
            try:
                # 内存中 resize + base64，不保存文件
                from PIL import Image
                import io
                img = Image.open(frame_path).convert("RGB")
                img.thumbnail((448, 448), Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=95)
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {e}")
                return None

        futures = [self.executor.submit(process_single_frame, path) for path in frame_paths]
        image_parts = []
        for future in futures:
            try:
                result = future.result(timeout=120)
                if result:
                    image_parts.append(result)
            except TimeoutError:
                logger.error(f"Frame processing timeout, skipping...")
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue
        return image_parts

    async def infer_async(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """异步推理"""
        if self.model_type == "ollama":
            return await self.infer_with_ollama_async(messages)
        elif self.model_type == "openai":
            return await self.infer_with_openai_async(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def infer_with_ollama_async(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """异步ollama推理"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }

            async with self.session.post(
                self.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    return None

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    async def infer_with_openai_async(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """异步OpenAI推理"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                timeout=180.0,  
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    
    def infer(self, frame_paths: List[str], title: str, annotation: str = "") -> Optional[Dict[str, Any]]:
        """同步推理接口"""
        try:
            # 预处理图片
            image_parts = []
            for frame_path in frame_paths:
                try:
                    # 对于Qwen2.5-VL，使用base64格式
                    with open(frame_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # 检测图片类型
                    mime_type = "image/jpeg"
                    if frame_path.lower().endswith('.png'):
                        mime_type = "image/png"
                    
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    })
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {e}")
                    continue

            if not image_parts:
                return None

            # 根据标志选择不同的prompt
            if self.use_vot_prompt:
                prompt = self.generate_vot_prompt(title)
            else:
                prompt = self.generate_basic_prompt(title)
            
            # 构建消息
            if self.model_type == "openai":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_parts
                        ]
                    }
                ]
            elif self.model_type == "ollama":
                # Ollama使用OpenAI格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_parts
                        ]
                    }
                ]
            # 进行推理
            if self.model_type == "ollama":
                response = self.infer_with_ollama(messages)
            elif self.model_type == "openai":
                response = self.infer_with_openai(messages)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            if response:
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content:
                    content = response.get('content', '')
                
                # 根据prompt类型解析响应
                if self.use_vot_prompt:
                    # VOT prompt解析
                    object_analysis = self.extract_section(content, "[Object Identification]:", "[Event Identification]:")
                    event_description = self.extract_section(content, "[Event Identification]:", "[Misinformation Prediction]:")
                    misinfo_prediction = self.extract_section(content, "[Misinformation Prediction]:", "")
                    
                    # 如果无法解析格式，尝试直接提取数字
                    if not misinfo_prediction:
                        misinfo_prediction = self.extract_prediction_from_text(content)
                    
                    # 确保最终预测是0或1
                    final_prediction = "0"
                    if misinfo_prediction and ('1' in misinfo_prediction or 'misinformation' in content.lower()):
                        final_prediction = "1"
                    elif misinfo_prediction and ('0' in misinfo_prediction or 'non-misinformation' in content.lower()):
                        final_prediction = "0"

                    result = {
                        'object_analysis': object_analysis,
                        'event_description': event_description,
                        'misinfo_prediction': misinfo_prediction,
                        'final_prediction': final_prediction
                    }
                else:
                    # 基础prompt解析
                    final_prediction = content.strip()
                    # 确保是0或1
                    if '1' in final_prediction:
                        final_prediction = "1"
                    elif '0' in final_prediction:
                        final_prediction = "0"
                    else:
                        # 如果无法解析，尝试提取数字
                        final_prediction = self.extract_prediction_from_text(content)
                        if not final_prediction:
                            final_prediction = "0"  # 默认值

                    result = {
                        'response': content,
                        'final_prediction': final_prediction
                    }
                
                return result

        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return None

    def infer_with_ollama(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """同步ollama推理"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    def infer_with_openai(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """同步OpenAI推理"""
        try:
            completion = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                timeout=180.0,
            )
            return completion.model_dump()

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    
    def run_inference(self, data_root,
                     frame_dir="frames", jsonl_file="HMVD.jsonl",
                     split="all", output_dir=None, max_videos=None,
                     save_interval=5, resume=True):
        """同步推理接口"""
        # 设置输出目录
        if output_dir is None:
            raise ValueError("output_dir must be specified")

        # 根据prompt类型设置不同的结果文件名
        prompt_type = "vot_prompt" if self.use_vot_prompt else "basic_prompt"
        results_file = os.path.join(output_dir, f'{prompt_type}_inference_results_{split}.json')

        # 加载已有结果
        existing_results = []
        processed_video_ids = set()
        if resume and os.path.exists(results_file):
            existing_results = self.load_results(results_file)
            processed_video_ids = {item['video_id'] for item in existing_results}
            logger.info(f"Resuming from {len(processed_video_ids)} processed videos")

        # 加载数据集
        dataset = self.load_dataset(
            data_root=data_root,
            jsonl_file=jsonl_file,
            frame_dir=frame_dir,
            split=split,
            processed_video_ids=processed_video_ids
        )

        if max_videos:
            dataset = dataset[:max_videos]

        prompt_name = "VOT prompt" if self.use_vot_prompt else "basic prompt"
        logger.info(f"Processing {len(dataset)} videos with {self.model_name} using {prompt_name}")

        # 存储结果
        results_dict = {item['video_id']: item for item in existing_results}

        # 处理视频
        for i, video_data in enumerate(tqdm(dataset, desc="Processing videos")):
            try:
                logger.info(f"Processing video {video_data['video_id']} ({len(video_data['frame_paths'])} frames)")

                result = self.infer(
                    video_data['frame_paths'], 
                    video_data['title'], 
                    video_data['annotation']
                )

                if result:
                    final_result = {
                        'video_id': video_data['video_id'],
                        'title': video_data['title'],
                        'annotation': video_data['annotation'],
                        'fake_type': video_data['fake_type'],
                        'label': video_data['label']
                    }
                    
                    # 根据prompt类型添加不同的字段
                    if self.use_vot_prompt:
                        final_result.update({
                            'object_analysis': result['object_analysis'],
                            'event_description': result['event_description'],
                            'misinfo_prediction': result['misinfo_prediction'],
                        })
                    else:
                        final_result.update({
                            'response': result['response'],
                        })
                    
                    final_result['final_prediction'] = result['final_prediction']
                    results_dict[video_data['video_id']] = final_result

                # 定期保存
                if save_interval > 0 and (i + 1) % save_interval == 0:
                    self.save_results(list(results_dict.values()), results_file)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing video {i}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # 保存最终结果
        final_results = list(results_dict.values())
        self.save_results(final_results, results_file)

        logger.info(f"Completed. Total processed: {len(final_results)}")

        if final_results:
            real_count = sum(1 for r in final_results if r['label'] == 1)
            fake_count = sum(1 for r in final_results if r['label'] == 0)
            predicted_fake = sum(1 for r in final_results if r['final_prediction'] == '1')
            predicted_real = sum(1 for r in final_results if r['final_prediction'] == '0')
            
            logger.info(f"Real videos: {real_count}, Fake videos: {fake_count}")
            logger.info(f"Predicted fake: {predicted_fake}, Predicted real: {predicted_real}")

        return final_results

    def load_results(self, results_file):
        """加载已有结果"""
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading results: {e}")
        return []

    def save_results(self, results, results_file):
        """保存结果"""
        try:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    async def process_batch_async(self, batch: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """异步处理一批视频"""
        tasks = [self.process_single_video_async(video_data) for video_data in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def run_inference_optimized(self,
                                    data_root,
                                    frame_dir="frames", jsonl_file="HMVD.jsonl",
                                    split="all", output_dir=None, max_videos=None,
                                    save_interval=5, resume=True):
        """优化后的异步批处理推理"""
        # 设置输出目录
        if output_dir is None:
            raise ValueError("output_dir must be specified")

        # 根据prompt类型设置不同的结果文件名
        prompt_type = "vot_prompt" if self.use_vot_prompt else "basic_prompt"
        results_file = os.path.join(output_dir, f'{prompt_type}_inference_results_{split}_optimized.json')

        # 加载已有结果
        existing_results = []
        processed_video_ids = set()
        if resume and os.path.exists(results_file):
            existing_results = self.load_results(results_file)
            processed_video_ids = {item['video_id'] for item in existing_results}
            logger.info(f"Resuming from {len(processed_video_ids)} processed videos")

        # 加载数据集
        dataset = self.load_dataset(
            data_root=data_root,
            jsonl_file=jsonl_file,
            frame_dir=frame_dir,
            split=split,
            processed_video_ids=processed_video_ids
        )

        if max_videos:
            dataset = dataset[:max_videos]

        prompt_name = "VOT prompt" if self.use_vot_prompt else "basic prompt"
        logger.info(f"Processing {len(dataset)} videos with {self.model_name} using {prompt_name} (batch_size={self.batch_size}, max_concurrent={self.max_concurrent})")

        # 存储结果
        results_dict = {item['video_id']: item for item in existing_results}

        # 分批处理
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} videos)")

            # 异步处理当前批次
            try:
                batch_results = await self.process_batch_async(batch)
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} processing failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

            # 更新结果
            for i, result in enumerate(batch_results):
                video_id = batch[i]['video_id'] if i < len(batch) else 'unknown'
                if isinstance(result, Exception):
                    logger.error(f"Video {video_id} processing error: {result}")
                elif result and not isinstance(result, Exception):
                    results_dict[result['video_id']] = result
                else:
                    logger.warning(f"Video {video_id} returned no result")

            # 定期保存
            if save_interval > 0 and (batch_idx + 1) % save_interval == 0:
                self.save_results(list(results_dict.values()), results_file)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 关闭会话
        if self.session:
            await self.session.close()

        # 保存最终结果
        final_results = list(results_dict.values())
        self.save_results(final_results, results_file)

        logger.info(f"Completed. Total processed: {len(final_results)}")

        if final_results:
            real_count = sum(1 for r in final_results if r['label'] == 1)
            fake_count = sum(1 for r in final_results if r['label'] == 0)
            predicted_fake = sum(1 for r in final_results if r['final_prediction'] == '1')
            predicted_real = sum(1 for r in final_results if r['final_prediction'] == '0')
            
            logger.info(f"Real videos: {real_count}, Fake videos: {fake_count}")
            logger.info(f"Predicted fake: {predicted_fake}, Predicted real: {predicted_real}")

        return final_results


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='openai',
                       choices=['ollama', 'openai'], help='模型类型')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--api_url', type=str, default=None, help='API地址')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--frame_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--jsonl_file', type=str, default='HMVD_repaired.jsonl')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'real', 'fake'])
    parser.add_argument('--max_videos', type=int, default=None,
                       help='最大处理视频数量')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存间隔')
    parser.add_argument('--resume', action='store_true',
                       help='从断点恢复')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='线程池最大工作线程数')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批处理大小')
    parser.add_argument('--cache_dir', type=str, default='image_cache',
                       help='图片缓存目录')
    parser.add_argument('--sync_mode', action='store_true',
                       help='使用同步推理模式（默认为异步模式）')
    parser.add_argument('--max_concurrent', type=int, default=3,
                       help='最大并发数')
    parser.add_argument('--vot_prompt', action='store_true',
                       help='使用视频思维链prompt推理')
    return parser


if __name__ == '__main__':
    args = get_args().parse_args()

    # 创建推理器
    inference = HMVDInference(
        model_type=args.model_type,
        model_name=args.model_name,
        api_url=args.api_url,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        max_concurrent=args.max_concurrent,
        use_vot_prompt=args.vot_prompt  # 传递参数
    )

    
    # 根据模式选择推理方式
    if args.sync_mode:
        logger.info("使用同步推理模式...")
        results = inference.run_inference(
            data_root=args.data_root,
            frame_dir=args.frame_dir,
            jsonl_file=args.jsonl_file,
            split=args.split,
            output_dir=args.output_dir,
            max_videos=args.max_videos,
            save_interval=args.save_interval,
            resume=args.resume
        )
    else:
        logger.info("使用异步推理模式...")
        # 运行异步推理
        async def main():
            results = await inference.run_inference_optimized(
                data_root=args.data_root,
                frame_dir=args.frame_dir,
                jsonl_file=args.jsonl_file,
                split=args.split,
                output_dir=args.output_dir,
                max_videos=args.max_videos,
                save_interval=args.save_interval,
                resume=args.resume
            )
            return results

        # 运行异步主函数
        results = asyncio.run(main())