#!/usr/bin/env python
"""
虚假视频检测工作流主脚本
支持数据集遍历和结果保存，并优化性能
"""
import os
import sys
import json
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import threading
from functools import lru_cache

# 处理导入问题，支持直接运行脚本和作为模块导入
try:
    from .workflow import kickoff, FakeVideoDetectorWorkflow
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from FakeAgent.workflow import kickoff, FakeVideoDetectorWorkflow

# 全局锁，用于线程安全的文件写入
file_lock = threading.Lock()

@lru_cache(maxsize=1)
def load_metadata_cached(jsonl_path: str) -> List[Dict]:
    """缓存加载HMVD元数据"""
    metadata = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            metadata.append(data)
    return metadata

def process_single_video_optimized(video_data: Dict, output_dir: Path, workflow_pool: List[FakeVideoDetectorWorkflow]) -> Dict:
    """优化版：处理单个视频（使用工作流池）"""
    video_id = video_data['video_id']
    title = video_data['title']
    annotation = video_data.get('annotation', '')
    fake_type = video_data.get('fake_type', '')
    
    # 构建视频路径
    video_path = f"/data/yyf/dataset/fakevideo/HMVD/{video_id}.mp4"
    video_frames_path = f"/root/siton-tmp/dataset/frames_noresize/{video_id}"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_frames_path):
        return {
            'video_id': video_id,
            'status': 'failed',
            'error': f'视频文件不存在: {video_path}',
            'title': title,
            'annotation': annotation,
            'fake_type': fake_type
        }
    
    try:
        # 从工作流池获取工作流实例
        workflow = workflow_pool.pop() if workflow_pool else FakeVideoDetectorWorkflow()
        
        start_time = time.time()
        result = workflow.run_workflow(video_path, title)
        processing_time = time.time() - start_time
        
        # 将工作流实例返回到池中
        workflow_pool.append(workflow)
        
        if result and not result.get("error"):
            # 异步保存智能体结果
            save_agent_results_async(result, output_dir, video_id)
            
            return {
                'video_id': video_id,
                'status': 'success',
                'title': title,
                'annotation': annotation,
                'fake_type': fake_type,
                'result': result,
                'processing_time': processing_time
            }
        else:
            error_msg = result.get("error", "未知错误") if result else "工作流执行失败"
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': error_msg,
                'title': title,
                'annotation': annotation,
                'fake_type': fake_type,
                'processing_time': processing_time
            }
    except Exception as e:
        return {
            'video_id': video_id,
            'status': 'failed',
            'error': str(e),
            'title': title,
            'annotation': annotation,
            'fake_type': fake_type,
            'processing_time': 0
        }

def save_agent_results_async(result: Dict, output_dir: Path, video_id: str):
    """异步保存智能体结果（使用线程锁确保安全）"""
    def save_async():
        agents_dir = output_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        
        agent_results = {
            'consistency_analysis': result.get('consistency_analysis', ''),
            'ai_detection': result.get('ai_detection', ''),
            'offensive_language_detection': result.get('offensive_language_detection', ''),
            'fact_checking': result.get('fact_checking', ''),
            'external_evidence': result.get('external_evidence', ''),
            'suspicious_segments': result.get('suspicious_segments', ''),
            'analysis': result.get('analysis', ''),
        }
        
        for agent_name, agent_result in agent_results.items():
            agent_file = agents_dir / f"{agent_name}.json"
            
            with file_lock:
                # 加载现有结果或创建新文件
                if agent_file.exists():
                    try:
                        with open(agent_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = {}
                else:
                    existing_data = {}
                
                # 添加当前视频的结果
                existing_data[video_id] = {
                    'result': agent_result,
                    'timestamp': str(Path(result.get('timestamp', '')))
                }
                
                # 保存更新后的结果
                with open(agent_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # 在后台线程中执行保存操作
    threading.Thread(target=save_async, daemon=True).start()

def process_dataset_parallel(jsonl_path: str, output_dir: str, max_videos: Optional[int] = None, 
                           skip_existing: bool = True, max_workers: int = 8):
    """优化版：并行处理整个数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载元数据（使用缓存）
    metadata = load_metadata_cached(jsonl_path)
    if max_videos:
        metadata = metadata[:max_videos]
    
    print(f"开始处理 {len(metadata)} 个视频，使用 {max_workers} 个工作线程...")
    
    # 加载现有结果（如果存在）
    results_file = output_path / "detection_results.json"
    existing_results = {}
    if skip_existing and results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = {r['video_id']: r for r in existing_data if 'error' not in r.keys() and 'error' not in json.loads(r['result']['analysis']).keys()}
            print(f"已加载 {len(existing_results)} 个现有结果")
        except Exception as e:
            print(f"加载现有结果失败: {e}")
    
    # 过滤掉已处理的视频
    videos_to_process = [v for v in metadata if not (skip_existing and v['video_id'] in existing_results)]
    print(f"需要处理 {len(videos_to_process)} 个新视频")
    
    # 创建工作流池（避免重复创建实例）
    workflow_pool = [FakeVideoDetectorWorkflow() for _ in range(max_workers)]
    
    results = list(existing_results.values())  # 包含现有结果
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(process_single_video_optimized, video_data, output_path, workflow_pool): video_data
            for video_data in videos_to_process
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(videos_to_process), desc="处理视频") as pbar:
            for future in concurrent.futures.as_completed(future_to_video):
                video_data = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 实时保存进度（每5个视频保存一次）
                    if len(results) % 10 == 0:
                        save_results_threadsafe(results, results_file)
                        pbar.set_postfix({"已保存": len(results)})
                    
                except Exception as e:
                    print(f"处理视频 {video_data['video_id']} 时出错: {e}")
                    results.append({
                        'video_id': video_data['video_id'],
                        'status': 'failed',
                        'error': str(e),
                        'title': video_data.get('title', ''),
                        'annotation': video_data.get('annotation', ''),
                        'fake_type': video_data.get('fake_type', ''),
                        'processing_time': 0
                    })
                
                pbar.update(1)
    
    # 最终保存
    save_results_threadsafe(results, results_file)
    
    # 统计处理时间
    processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"平均处理时间: {avg_time:.2f} 秒/视频")
    
    return results

def save_results_threadsafe(results: List[Dict], output_path: Path):
    """线程安全地保存结果"""
    with file_lock:
        try:
            # 创建临时文件，原子性写入
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 重命名临时文件为目标文件
            temp_path.replace(output_path)
            
        except Exception as e:
            print(f"保存结果失败: {e}")

def process_single_video_simple(video_path: str, video_title: str = "") -> Dict:
    """简化版：处理单个视频（用于快速测试）"""
    print(f"处理单个视频: {video_path}")
    
    if not os.path.exists(video_path):
        return {'status': 'failed', 'error': f'视频文件不存在: {video_path}'}
    
    try:
        start_time = time.time()
        result = kickoff(video_path, video_title)
        processing_time = time.time() - start_time
        
        if result:
            return {
                'status': 'success',
                'result': result,
                'processing_time': processing_time
            }
        else:
            return {'status': 'failed', 'error': '工作流执行失败'}
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def parse_args():
    """解析命令行参数（优化版）"""
    parser = argparse.ArgumentParser(description="虚假视频检测工作流（优化版）")
    parser.add_argument("--jsonl_path",
                        default=None,
                        help="HMVD.jsonl文件路径")
    parser.add_argument("--output_dir",
                        default=None,
                        help="检测结果输出目录")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="最大处理视频数量（用于测试）")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已处理的视频")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="不跳过已处理的视频")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并行处理的工作线程数")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 处理整个数据集（并行）
    print("开始并行处理数据集...")
    start_time = time.time()
    results = process_dataset_parallel(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        skip_existing=args.skip_existing,
        max_workers=args.max_workers
    )
    total_time = time.time() - start_time
    
    # 统计结果
    total = len(results)
    success = len([r for r in results if r['status'] == 'success'])
    failed = total - success
    
    print("\n处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"总计: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"成功率: {success / total:.2%}" if total > 0 else "成功率: 0%")
    print(f"平均速度: {total_time/total:.2f} 秒/视频" if total > 0 else "平均速度: N/A")
    print(f"结果目录: {args.output_dir}")