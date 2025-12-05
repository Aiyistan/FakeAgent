import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflows.detector import FakeVideoDetectorWorkflow, kickoff, batch_kickoff


def test_single_video_detection():
    """测试单个视频检测"""
    print("=== 测试单个视频检测 ===")
    
    # 测试视频路径（请替换为实际的视频文件路径）
    video_path = "/root/siton-tmp/dataset/HMVD/zZB5xaS5SNA.mp4"
    video_title = "Space X’s First Civilian Spacewalk Undertaken By Polaris Dawn Crew | 10 News First"
    
    if os.path.exists(video_path):

        try:
            result = kickoff(
                video_path=video_path,  
                video_title=video_title,
                use_preprocessing=True,
                audio_model_dir="/root/siton-tmp/models/SenseVoiceSmall/"
            )
            
            print("检测结果:")
            print(f"- 视频标题: {result.get('video_title', '无')}")
            print(f"- 转录内容: {result.get('transcription', '无')[:50]}...")
            print(f"- 帧描述数量: {len(result.get('frame_descriptions', []))}")
            print(f"- 一致性分析: {result.get('consistency_analysis', '无')[:50]}...")
            print(f"- AI检测: {result.get('ai_detection', '无')[:50]}...")
            print(f"- 事实核查: {result.get('fact_checking', '无')[:50]}...")
            print(f"- 可疑片段: {result.get('suspicious_segments', '无')[:50]}...")
            print(f"- 最终分析: {result.get('analysis', '无')[:50]}...")
            
            return True
            
        except Exception as e:
            print(f"测试失败: {e}")
            return False
            


if __name__ == "__main__":
    print("开始运行 FakeAgent 检测器测试...")
    
    # 运行测试
    test1_result = test_single_video_detection()
