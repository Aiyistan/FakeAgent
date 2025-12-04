#!/usr/bin/env python3
"""
InternVL3-8B 推理模块
用于对视频提取的8帧图片进行推理分析
"""

import os
from typing import List, Union, Optional
from pathlib import Path

import json
import logging
from typing import Dict
import torch
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse


class InternVL3Inference:
    """InternVL3-8B 推理类"""
    
    def __init__(
        self,
        model_name: str,
        session_len: int,
        max_batch_size: int,
        devices: str,
        tp: int
    ):
        self.model_name = model_name
        self.session_len = session_len
        self.max_batch_size = max_batch_size
        self.devices = devices
        self.tp = tp
        
        # 初始化推理管道
        self.pipe = self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> pipeline:
        """初始化 LMDeploy 推理管道"""
        try:
            # 设置环境变量以控制GPU使用
            if self.devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
                print(f"设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.model_name):
                raise FileNotFoundError(f"模型路径不存在: {self.model_name}")
            
            print(f"模型路径: {self.model_name}")
            
            backend_config = TurbomindEngineConfig(
                session_len=self.session_len,
                max_batch_size=self.max_batch_size,
                tp=self.tp
            )
            
            print(f"初始化pipeline，tp={self.tp}, session_len={self.session_len}")
            
            pipe = pipeline(
                self.model_name,
                backend_config=backend_config,
                chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
            )
            
            print("Pipeline初始化成功")
            return pipe
            
        except Exception as e:
            print(f"Pipeline初始化失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            
            raise e

    def batch_inference(
        self,
        image_paths: List[Union[str, Path]],
        **generation_kwargs
    ) -> List[str]:
        # 构造提示词
        prompts = [('describe this image:', load_image(image_path)) for image_path in image_paths]
        
        # 设置默认生成参数
        default_kwargs = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 512,
            "topk": 40
        }
        default_kwargs.update(generation_kwargs)
        results = self.pipe(prompts, **default_kwargs)
        results = [result.text for result in results]
        return results

class HMVDInferenceProcessor:
    """HMVD数据集推理处理器"""
    
    def __init__(
        self,
        jsonl_path: str,
        frames_base_dir: str,
        output_dir: str,
        model_path: str,
        session_len: int,
        max_batch_size: int,
        tp: int,
        devices: str
    ):
        self.jsonl_path = Path(jsonl_path)
        self.frames_base_dir = Path(frames_base_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.session_len = session_len
        self.max_batch_size = max_batch_size
        self.tp = tp
        self.devices = devices
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化推理器
        self.inferencer = InternVL3Inference(
            model_name=self.model_path,
            session_len=self.session_len,
            max_batch_size=self.max_batch_size,
            tp=self.tp,
            devices=self.devices
        )
        
    def _setup_logging(self):
        """设置日志配置"""
        log_file = self.output_dir / "inference.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_metadata(self) -> List[Dict]:
        """加载HMVD元数据"""
        metadata = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata.append(data)
        return metadata
    
    def get_frame_paths(self, video_id: str) -> List[str]:
        frames_dir = self.frames_base_dir / video_id
        if not frames_dir.exists():
            return []
            
        # frams_dir下所有文件为8帧图片
        frame_paths = os.listdir(frames_dir)
        frame_paths = [frames_dir / path for path in frame_paths]
            
        return [str(path) for path in frame_paths]
    
    def process_single_video(
        self, 
        video_id: str, 
        **generation_kwargs
    ) -> Dict:
        # 获取帧图片路径
        frame_paths = self.get_frame_paths(video_id)
        
        if not frame_paths:
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': f'未找到帧图片: {video_id}',
                'analysis': '',
                'frame_count': 0
            }
        
        try:
            responses = self.inferencer.batch_inference(
                frame_paths,
                **generation_kwargs
            )
            
            return {
                'video_id': video_id,
                'status': 'success',
                'frame_descriptions': responses,
                'frames': frame_paths
            }
            
        except Exception as e:
            self.logger.error(f"处理视频 {video_id} 时出错: {e}")
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def process_all_videos(
        self,
        max_videos: Optional[int] = None,
        skip_existing: bool = True,
        **generation_kwargs
    ) -> List[Dict]:
        """
        处理所有视频
        """
        metadata = self.load_metadata()
        
        if max_videos:
            metadata = metadata[:max_videos]
            
        self.logger.info(f"开始处理 {len(metadata)} 个视频...")
        
        # 加载现有结果（如果存在）
        results_file = self.output_dir / "frame_descriptions.json"
        existing_results = {}
        
        if skip_existing and results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_results = {r['video_id']: r for r in existing_data}
                self.logger.info(f"已加载 {len(existing_results)} 个现有结果")
            except Exception as e:
                self.logger.warning(f"加载现有结果失败: {e}")
        
        results = []
        
        # 处理每个视频
        for video_data in tqdm(metadata, desc="处理视频"):
            video_id = video_data['video_id']
            title = video_data['title']
            
            # 检查是否已处理
            if skip_existing and video_id in existing_results:
                results.append(existing_results[video_id])
                continue
                
            # 处理视频，传入title
            result = self.process_single_video(
                video_id, 
                **generation_kwargs
            )
            
            # 添加元数据信息
            result.update({
                'title': title,
                'annotation': video_data.get('annotation', ''),
                'fake_type': video_data.get('fake_type', '')
            })
            
            results.append(result)
            
            # 实时保存进度
            if len(results) % 10 == 0:
                self.save_results(results, results_file)
                self.logger.info(f"已处理 {len(results)} 个视频")
        
        # 最终保存
        self.save_results(results, results_file)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: Path):
        """保存推理结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """生成处理摘要"""
        total = len(results)
        success = len([r for r in results if r['status'] == 'success'])
        failed = total - success
        
        summary = {
            'total_videos': total,
            'successful': success,
            'failed': failed,
            'success_rate': success / total if total > 0 else 0,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        }
        
        # 保存摘要
        summary_file = self.output_dir / "frame_inference_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存摘要失败: {e}")
            
        return summary
    
    def save_failed_videos(self, results: List[Dict]):
        """保存失败视频列表以便重新处理"""
        failed_videos = [r for r in results if r['status'] == 'failed']
        
        if failed_videos:
            failed_file = self.output_dir / "failed_frame_inference_videos.json"
            try:
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_videos, f, ensure_ascii=False, indent=2)
                self.logger.info(f"失败视频列表已保存到: {failed_file}")
            except Exception as e:
                self.logger.error(f"保存失败视频列表失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HMVD数据集InternVL3视频帧推理脚本")
    parser.add_argument("--jsonl_path", 
                       default="/data/yyf/projects/video/HMVD/data/HMVD.jsonl",
                       help="HMVD.jsonl文件路径")
    parser.add_argument("--frames_dir", 
                       default="/data/yyf/projects/video/HMVD/data/frames",
                       help="帧图片基础目录")
    parser.add_argument("--output_dir", 
                       default="/data/yyf/projects/video/HMVD/data/frame_inference_results",
                       help="推理结果输出目录")
    parser.add_argument("--model_path", 
                       default="/data/yyf/model/InternVL2_5-8B/",
                       help="InternVL2_5-8B 模型路径")
    parser.add_argument("--tp", type=int, default=2,
                       help="tensor parallelism数量")
    parser.add_argument("--devices", type=str, default='0,1',
                       help="指定GPU设备，如 --devices '0, 1'")
    parser.add_argument("--session_len", type=int, default=int(8192),
                       help="会话长度")
    parser.add_argument("--max_batch_size", type=int, default=8,
                       help="最大批处理大小")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="最大处理视频数量（用于测试）")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.8,
                       help="top-p采样")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="最大新生成token数")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="跳过已处理的视频")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                       help="不跳过已处理的视频")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = HMVDInferenceProcessor(
        jsonl_path=args.jsonl_path,
        frames_base_dir=args.frames_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        session_len=args.session_len,
        max_batch_size=args.max_batch_size,
        tp=args.tp,
        devices=args.devices
    )
    
    # 处理所有视频（使用默认prompt）
    results = processor.process_all_videos(
        max_videos=args.max_videos,
        skip_existing=args.skip_existing,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    # 生成摘要
    summary = processor.generate_summary(results)
    
    # 保存失败视频列表
    processor.save_failed_videos(results)
    
    # 打印摘要
    print(f"\n推理完成！")
    print(f"总计: {summary['total_videos']}")
    print(f"成功: {summary['successful']}")
    print(f"失败: {summary['failed']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()