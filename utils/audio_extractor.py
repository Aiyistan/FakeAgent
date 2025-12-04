#!/usr/bin/env python3
"""
音频提取和转录模块
基于现有HMVD预处理框架，提供音频提取和ASR转录功能
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from moviepy.video import fx as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



class AudioExtractor:
    """音频提取器 - 从视频中提取音频并转录"""
    
    def __init__(self, model_dir: str, device: int, video_base_dir: str, audio_output_dir: str):
        """
        初始化音频提取器
        
        Args:
            video_base_dir: 视频文件基础目录
            audio_output_dir: 音频输出目录
        """
        self.video_base_dir = Path(video_base_dir)
        self.audio_output_dir = Path(audio_output_dir)
        
        # 创建输出目录
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化转录器
        self.model_dir = model_dir
        self.device = f"cuda:{device}"
        self.model = AutoModel(
            model=self.model_dir, 
            # trust_remote_code=True,
            # remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device
        )
    
    def extract_audio_from_video(self, video_path:str):
        """
        从视频提取完整音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            音频文件路径
        """
        video_name = Path(video_path).stem
        audio_path = self.audio_output_dir / f"{video_name}.mp3"
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(str(audio_path))
        
        return audio_path
    
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> Dict:
        """
        对音频文件进行ASR转录
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录结果字典
        """
        res = self.model.generate(
            input=audio_path,
            cache={},
            language=language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text


class AudioPreprocessor:
    """数据集音频预处理器"""
    
    def __init__(self, model_dir: str, device: int, jsonl_path: str, video_base_dir: str, audio_output_dir: str):
        """
        初始化音频预处理器
        
        Args:
            model_dir: 模型路径,
            device: int,
            jsonl_path: HMVD.jsonl文件路径
            video_base_dir: 视频文件基础目录
            audio_output_dir: 音频输出目录
        """
        self.model_dir = model_dir
        self.device = device
        self.jsonl_path = Path(jsonl_path)
        self.video_base_dir = Path(video_base_dir)
        self.audio_output_dir = Path(audio_output_dir)
        
        # 初始化音频提取器
        self.audio_extractor = AudioExtractor(
            model_dir=self.model_dir,
            device=self.device,
            video_base_dir=str(self.video_base_dir),
            audio_output_dir=str(self.audio_output_dir)
        )
    
    def load_metadata(self) -> List[Dict]:
        """加载HMVD元数据"""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def process_single_video(self, video_id: str, language: str = "auto") -> Dict:
        """
        处理单个视频的音频提取和转录
        
        Args:
            video_id: 视频ID
            segment_length: 音频片段长度（秒）
            
        Returns:
            处理结果字典
        """
        try:
            # 构建视频文件路径
            video_path = self.video_base_dir / f"{video_id}.mp4"
            if not video_path.exists():
                return {
                    'video_id': video_id,
                    'status': 'failed',
                    'error': f'视频文件不存在: {video_id}'
                }
            
            # 提取音频
            audio_path = self.audio_extractor.extract_audio_from_video(
                str(video_path)
            )
            
            # 转录音频
            transcript = self.audio_extractor.transcribe_audio(
                str(audio_path),
                language=language
            )
            
            return {
                'video_id': video_id,
                'status': 'success',
                'transcript': transcript
            }
            
        except Exception as e:
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
                'transcript': '无'
            }
    
    def process_all_videos(self, language: str = "auto", max_videos: Optional[int] = None, log_path: str = "audio_processing_log.json") -> List[Dict]:
        """
        处理所有视频的音频提取和转录
        
        Args:
            language: 转录语言
            max_videos: 最大处理视频数量（用于测试）
            log_path: 日志文件路径

        Returns:
            处理结果列表
        """
        metadata = self.load_metadata()
        results = []
        
        if max_videos:
            metadata = metadata[:max_videos]
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_video_ids = [r['video_id'] for r in results if r['status'] == 'success']
            processed_videos = [r for r in results if r['status'] == 'success']
        else:
            processed_video_ids = []
            processed_videos = []
        
        
        print(f"开始处理 {len(metadata)} 个视频的音频...")
        
        for video_data in tqdm(metadata, desc="处理音频"):
            video_id = video_data['video_id']
            if video_id in processed_video_ids:
                continue
            print(f"处理视频 {video_id}")
            result = self.process_single_video(video_id, language)
            
            # 添加元数据信息
            result.update({
                'title': video_data.get('title', ''),
                'annotation': video_data.get('annotation', ''),
                'fake_type': video_data.get('fake_type', '')
            })
            processed_videos.append(result)
            
            # results.append(result)
        
        return processed_videos
    
    def save_processing_log(self, results: List[Dict], log_path: str):
        """保存处理日志"""
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """生成处理摘要"""
        total = len(results)
        success = len([r for r in results if r['status'] == 'success'])
        failed = total - success
        
        summary = {
            'total_videos': total,
            'successful': success,
            'failed': failed,
            'success_rate': success / total if total > 0 else 0
        }
        
        return summary


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HMVD数据集音频提取和转录工具")
    parser.add_argument("--model_dir", 
                       default="/data/yyf/model/SenseVoiceSmall",
                       help="模型路径")
    parser.add_argument("--device", 
                       type=int,
                       default=1,
                       help="设备ID")
    parser.add_argument("--jsonl_path", 
                       default="/data/yyf/projects/video/HMVD/data/HMVD.jsonl",
                       help="HMVD.jsonl文件路径")
    parser.add_argument("--video_base_dir", 
                       default="/data/yyf/dataset/fakevideo/HMVD",
                       help="视频文件目录")
    parser.add_argument("--audio_output_dir", 
                       default="/data/yyf/projects/video/HMVD/data/audio",
                       help="音频输出目录")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="最大处理视频数量（用于测试）")
    parser.add_argument("--language", default="auto",
                       help="语言设置")
    parser.add_argument("--log_path", 
                       default="/data/yyf/projects/video/HMVD/data/audio_processing_log.json",
                       help="处理日志保存路径")
    
    args = parser.parse_args()
    
    # 初始化预处理器
    preprocessor = AudioPreprocessor(
        model_dir=args.model_dir,
        device=args.device,
        jsonl_path=args.jsonl_path,
        video_base_dir=args.video_base_dir,
        audio_output_dir=args.audio_output_dir,
    )
    
    # 处理所有视频
    results = preprocessor.process_all_videos(
        max_videos=args.max_videos,
        language=args.language,
        log_path=args.log_path
    )
    
    # 保存处理日志
    preprocessor.save_processing_log(results, args.log_path)
    
    # 生成摘要
    summary = preprocessor.generate_summary(results)
    
    print(f"使用语言: {args.language}")
    print(f"总计: {summary['total_videos']}")
    print(f"成功: {summary['successful']}")
    print(f"失败: {summary['failed']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"日志已保存到: {args.log_path}")


if __name__ == "__main__":
    main()
