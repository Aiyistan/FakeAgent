#!/usr/bin/env python3
"""
HMVD数据集视频帧提取器
从HMVD数据集中的视频中提取8帧关键帧用于后续分析
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Optional
from PIL import Image
import subprocess

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIE_PY_AVAILABLE = True
except ImportError:
    MOVIE_PY_AVAILABLE = False
    print("警告: MoviePy未安装，请先安装moviepy")


class VideoFrameExtractor:
    """视频帧提取器
    
    支持两种帧提取方法：
    1. 质量感知提取：智能选择高质量帧，时间分布均匀
    2. 简单均匀采样：快速均匀采样，无质量优化
    
    性能说明：
    - 质量感知提取：
        * 使用OpenCV时：约1-3秒/视频（推荐）
        * 使用numpy时：约5-15秒/视频（无OpenCV依赖）
    - 简单均匀采样：约2秒/视频
    
    自动检测OpenCV，优先使用以获得最佳性能
    """
    """视频帧提取器，专门用于从视频中提取8帧关键帧"""
    
    def __init__(self, video_dir: str, output_dir: str, num_frames: int = 8):
        """
        初始化视频帧提取器
        
        Args:
            video_dir: 视频文件所在目录
            output_dir: 帧图片输出目录
            num_frames: 要提取的帧数，默认为8
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames_uniform(self, video_path: str, output_prefix: str, quality_aware: bool = True) -> Tuple[List[str], List[float]]:
        """
        均匀采样提取帧，可选择质量感知或简单均匀采样
        
        Args:
            video_path: 视频文件路径
            output_prefix: 输出文件前缀
            quality_aware: 是否使用质量感知提取，True为质量感知，False为简单均匀采样
            
        Returns:
            tuple: (提取的帧文件路径列表, 抽帧时间点列表)
        """
        if not MOVIE_PY_AVAILABLE:
            raise RuntimeError("MoviePy不可用，请先安装moviepy")
            
        if quality_aware:
            return self._extract_frames_quality_aware(video_path, output_prefix)
        else:
            return self._extract_frames_simple_uniform(video_path, output_prefix)

    def _extract_frames_simple_uniform(self, video_path: str, output_prefix: str) -> tuple[List[str], List[float]]:
        """
        使用MoviePy进行简单均匀采样提取帧
        
        Args:
            video_path: 视频文件路径
            output_prefix: 输出文件前缀
            
        Returns:
            tuple: (提取的帧文件路径列表, 抽帧时间点列表)
        """
        extracted_paths = []
        frame_times_list = []
        
        with VideoFileClip(video_path) as video:
            total_duration = video.duration
            
            # 简单均匀采样时间点
            frame_times = np.linspace(0.1, total_duration - 0.1, self.num_frames)
            
            for i, t in enumerate(frame_times):
                try:
                    # 提取帧
                    frame = video.get_frame(t)
                    
                    # 转换为PIL Image并调整大小
                    pil_image = Image.fromarray(frame.astype('uint8'))
                    # pil_image = pil_image.resize((1280, 720))
                    
                    # 保存图片
                    output_path = self.output_dir / f"{output_prefix}_frame_{i+1:03d}.jpg"
                    pil_image.save(str(output_path), 'JPEG', quality=95)
                    extracted_paths.append(str(output_path))
                    # 保留三位小数
                    frame_times_list.append(round(t, 3))
                    
                except Exception as e:
                    # 如果当前时间点失败，尝试附近的时间点
                    search_range = 5.0  # 搜索范围5秒
                    step = 0.1  # 步长0.1秒
                    
                    found = False
                    for offset in np.arange(step, search_range, step):
                        for direction in [-1, 1]:  # 先向后搜索，再向前搜索
                            try_time = t + offset * direction
                            if 0 <= try_time <= total_duration:
                                try:
                                    frame = video.get_frame(try_time)
                                    pil_image = Image.fromarray(frame.astype('uint8'))
                                    # pil_image = pil_image.resize((1280, 720))
                                    
                                    output_path = self.output_dir / f"{output_prefix}_frame_{i+1:03d}.jpg"
                                    pil_image.save(str(output_path), 'JPEG', quality=95)
                                    extracted_paths.append(str(output_path))
                                    # 保留三位小数
                                    frame_times_list.append(round(try_time, 3))
                                    found = True
                                    break
                                except:
                                    continue
                        if found:
                            break
                    
                    if not found:
                        # 尝试提取最后一帧
                        try:
                            last_frame_time = max(0, total_duration - 0.1)
                            frame = video.get_frame(last_frame_time)
                            pil_image = Image.fromarray(frame.astype('uint8'))
                            # pil_image = pil_image.resize((1280, 720))
                            
                            output_path = self.output_dir / f"{output_prefix}_frame_{i+1:03d}.jpg"
                            pil_image.save(str(output_path), 'JPEG', quality=95)
                            extracted_paths.append(str(output_path))
                            # 保留三位小数
                            frame_times_list.append(round(last_frame_time, 3))
                        except:
                            raise RuntimeError(f"无法从视频 {video_path} 中提取第 {i+1} 帧")
        print("简单均匀采样抽帧时间点:", frame_times_list)                    
        return extracted_paths, frame_times_list

    def _extract_frames_quality_aware(self, video_path: str, output_prefix: str) -> Tuple[List[str], List[float]]:
        """优化的质量感知帧提取"""
        extracted_paths = []
        frame_times_list = []
        
        try:
            video = VideoFileClip(str(video_path))
            total_duration = video.duration
            fps = video.fps if hasattr(video, 'fps') else 25
            
            # 大幅减少候选帧数量，提高性能
            candidate_count = self.num_frames * 2  # 减少到2倍候选帧
            candidate_times = np.linspace(0.1, total_duration - 0.1, candidate_count)
            
            # 批量评估候选帧质量
            frame_scores = []
            for t in candidate_times:
                try:
                    frame = video.get_frame(t)
                    score = self._calculate_frame_quality_fast(frame)
                    frame_scores.append((t, score, frame))
                except:
                    continue
            
            if not frame_scores:
                raise RuntimeError(f"无法从视频 {video_path} 中提取任何有效帧")
            
            # 按质量分数排序，选择最好的帧
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 确保时间分布均匀
            selected_frames = []
            time_segments = np.linspace(0, total_duration, self.num_frames + 1)
            
            for i in range(self.num_frames):
                segment_start = time_segments[i]
                segment_end = time_segments[i + 1]
                
                # 在当前时间段内选择质量最好的帧
                segment_frames = [(t, score, frame) for t, score, frame in frame_scores 
                                if segment_start <= t <= segment_end]
                
                if segment_frames:
                    best_frame = max(segment_frames, key=lambda x: x[1])
                    selected_frames.append(best_frame)
                else:
                    # 如果没有候选帧，使用最近的高质量帧
                    target_time = (segment_start + segment_end) / 2
                    closest_frame = min(frame_scores, key=lambda x: abs(x[0] - target_time))
                    selected_frames.append(closest_frame)
            
            # 按时间排序并保存
            selected_frames.sort(key=lambda x: x[0])
            
            for i, (t, score, frame) in enumerate(selected_frames):
                # 转换为PIL Image并调整大小
                pil_image = Image.fromarray(frame.astype('uint8'))
                # pil_image = pil_image.resize((1280, 720))
                
                # 保存图片
                output_path = self.output_dir / f"{output_prefix}_frame_{i+1:03d}.jpg"
                pil_image.save(str(output_path), 'JPEG', quality=95)
                extracted_paths.append(str(output_path))
                frame_times_list.append(round(t, 3))
                
        except Exception as e:
            raise RuntimeError(f"处理视频 {video_path} 时出错: {str(e)}")

        print("质量感知抽帧时间点:", frame_times_list)     
        return extracted_paths, frame_times_list
    
    def _calculate_frame_quality_fast(self, frame: np.ndarray) -> float:
        """快速计算帧质量分数，优先使用OpenCV"""
        try:
            import cv2
            # 使用OpenCV进行快速质量评估
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # OpenCV的拉普拉斯边缘检测（极快）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # 对比度
            contrast = gray.std()
            
            # 熵（可选，OpenCV实现）
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist[hist > 0]
            entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum()))
            
            quality_score = sharpness * 0.4 + contrast * 0.3 + entropy * 0.3
            return float(quality_score)
            
        except ImportError:
            # 回退到numpy实现（无OpenCV依赖）
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
            
            # 使用简单的方差作为清晰度指标
            sharpness = np.var(gray)
            contrast = np.std(gray)
            
            # 简单的质量分数组合
            quality_score = sharpness * 0.6 + contrast * 0.4
            return quality_score


class HMVDPreprocessor:
    """HMVD数据集预处理器"""
    
    def __init__(self, jsonl_path: str, video_base_dir: str, output_base_dir: str):
        """
        初始化HMVD预处理器
        
        Args:
            jsonl_path: HMVD.jsonl文件路径
            video_base_dir: 视频文件基础目录
            output_base_dir: 输出基础目录
        """
        self.jsonl_path = Path(jsonl_path)
        self.video_base_dir = Path(video_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.frame_extractor = VideoFrameExtractor(str(self.video_base_dir), str(self.output_base_dir))
        
    def load_metadata(self) -> List[Dict]:
        """加载HMVD元数据"""
        metadata = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata.append(data)
        return metadata
        
    def process_single_video(self, video_id: str, method: str = 'uniform', quality_aware: bool = True) -> Dict:
        """
        处理单个视频
        
        Args:
            video_id: 视频ID
            method: 帧提取方法 ('uniform' 或 'quality_aware')
            quality_aware: 是否使用质量感知提取（仅对uniform方法有效）
            
        Returns:
            处理结果信息
        """
        video_path = self.video_base_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': f'视频文件不存在: {video_path}',
                'frames': [],
                'frame_times': []
            }
            
        # 创建视频专属输出目录
        video_output_dir = self.output_base_dir / video_id
        video_output_dir.mkdir(exist_ok=True)
        
        # 更新帧提取器的输出目录
        self.frame_extractor.output_dir = video_output_dir
        
        try:
            if method == 'uniform':
                frames, frame_times = self.frame_extractor.extract_frames_uniform(
                    str(video_path), video_id, quality_aware=quality_aware
                )
            else:
                frames, frame_times = self.frame_extractor.extract_frames_uniform(
                    str(video_path), video_id, quality_aware=True
                )
                
            return {
                'video_id': video_id,
                'status': 'success',
                'method': method,
                'quality_aware': quality_aware,
                'frame_count': len(frames),
                'frames': frames,
                'frame_times': frame_times,
                'video_path': str(video_path)
            }
            
        except Exception as e:
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
                'frames': [],
                'frame_times': []
            }
            
    def process_all_videos(self, method: str = 'uniform', quality_aware: bool = True, max_videos: Optional[int] = None) -> List[Dict]:
        """
        处理所有视频
        
        Args:
            method: 帧提取方法
            quality_aware: 是否使用质量感知提取
            max_videos: 最大处理视频数量（用于测试）
            
        Returns:
            处理结果列表
        """
        metadata = self.load_metadata()
        results = []
        
        if max_videos:
            metadata = metadata[:max_videos]
            
        print(f"开始处理 {len(metadata)} 个视频...")
        
        for video_data in tqdm(metadata, desc="处理视频"):
            video_id = video_data['video_id']
            result = self.process_single_video(video_id, method, quality_aware)
            
            # 添加元数据信息
            result.update({
                'title': video_data.get('title', ''),
                'annotation': video_data.get('annotation', ''),
                'fake_type': video_data.get('fake_type', '')
            })
            
            results.append(result)
            
        return results
        
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
            'success_rate': success / total if total > 0 else 0,
            'method_used': results[0]['method'] if results else 'unknown'
        }
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HMVD数据集视频帧提取器")
    parser.add_argument("--jsonl_path", default="/data/yyf/projects/video/HMVD/data/HMVD.jsonl", 
                       help="HMVD.jsonl文件路径")
    parser.add_argument("--video_dir", default="/data/yyf/dataset/fakevideo/HMVD", 
                       help="视频文件目录")
    parser.add_argument("--output_dir", default="/data/yyf/projects/video/HMVD/data/frames_noresize", 
                       help="输出帧图片目录")
    parser.add_argument("--method", choices=['uniform'], 
                       default='uniform', help="帧提取方法")
    parser.add_argument("--quality_aware", action='store_true', default=True,
                       help="使用质量感知提取（默认启用）")
    parser.add_argument("--no_quality_aware", action='store_false', dest='quality_aware',
                       help="禁用质量感知提取，使用简单均匀采样")
    parser.add_argument("--max_videos", type=int, default=None, 
                       help="最大处理视频数量（用于测试）")
    parser.add_argument("--log_path", default="/data/yyf/projects/video/HMVD/data/processing_log2.json", 
                       help="处理日志保存路径")
    
    args = parser.parse_args()
    
    # 初始化预处理器
    preprocessor = HMVDPreprocessor(
        args.jsonl_path,
        args.video_dir,
        args.output_dir
    )
    
    # 处理所有视频
    results = preprocessor.process_all_videos(
        method=args.method,
        quality_aware=args.quality_aware,
        max_videos=args.max_videos
    )
    
    # 保存处理日志
    preprocessor.save_processing_log(results, args.log_path)
    
    # 生成摘要
    summary = preprocessor.generate_summary(results)
    
    print(f"\n处理完成！")
    print(f"总计: {summary['total_videos']}")
    print(f"成功: {summary['successful']}")
    print(f"失败: {summary['failed']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"使用质量感知: {'是' if args.quality_aware else '否'}")
    print(f"日志已保存到: {args.log_path}")


if __name__ == "__main__":
    main()