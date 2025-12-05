#!/usr/bin/env python
"""
FakeAgent 虚假视频检测 Gradio 界面
提供简洁的用户界面，用于上传视频并展示分析结果
"""
import os
import sys
import json
import gradio as gr
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflows.detector import kickoff

# 全局变量，用于存储最新的分析结果
latest_result = None

def analyze_video(video_file, video_title, use_preprocessing):
    """分析上传的视频"""
    global latest_result
    
    if video_file is None:
        return "请上传视频文件", "", "", "", "", "", "", "", ""
    
    try:
        # 调用检测函数
        start_time = time.time()
        result = kickoff(
            video_path=video_file,
            video_title=video_title,
            use_preprocessing=use_preprocessing,
            audio_model_dir="/root/siton-tmp/models/SenseVoiceSmall/",
            openai_api_base=os.getenv('MULTIMODAL_OPENAI_API_BASE'),
            openai_api_key=os.getenv('MULTIMODAL_OPENAI_API_KEY'),
            openai_model=os.getenv('MULTIMODAL_OPENAI_MODEL'),
            video_output_dir='./tmp/video_prerocess'
        )
        processing_time = time.time() - start_time
        
        # 存储结果
        latest_result = result
        
        # 提取各部分结果
        transcription = result.get('transcription', '无转录内容')
        frame_descriptions = result.get('frame_descriptions', [])
        frame_descriptions_str = '\n'.join(frame_descriptions) if frame_descriptions else '无帧描述'
        
        consistency_analysis = result.get('consistency_analysis', '无一致性分析')
        ai_detection = result.get('ai_detection', '无AI检测结果')
        offensive_language = result.get('offensive_language_detection', '无冒犯性语言检测结果')
        fact_checking = result.get('fact_checking', '无事实核查结果')
        external_evidence = result.get('external_evidence', '无外部证据')
        suspicious_segments = result.get('suspicious_segments', '无可疑片段定位')
        final_analysis = result.get('analysis', '无最终分析')
        
        
        # 返回处理信息和各部分结果
        status_msg = f"✅ 分析完成，耗时: {processing_time:.2f} 秒"
        
        return (
            status_msg,
            transcription,
            frame_descriptions_str,
            consistency_analysis,
            ai_detection,
            offensive_language,
            fact_checking,
            external_evidence,
            suspicious_segments,
            final_analysis
        )
        
    except Exception as e:
        error_msg = f"❌ 分析失败: {str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
        return (
            error_msg,
            "", "", "", "", "", "", "", ""
        )

def export_results():
    """导出最新的分析结果为JSON"""
    global latest_result
    
    if latest_result is None:
        return None, "没有可导出的结果，请先分析视频"
    
    try:
        # # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8') as temp_file:
            temp_path = temp_file.name
            json.dump(latest_result, temp_file, ensure_ascii=False, indent=2)
        
        return temp_path, "结果已导出"
    except Exception as e:
        return None, f"导出失败: {str(e)}"

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="FakeAgent 虚假视频检测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🕵️ FakeAgent 虚假视频检测系统")
        gr.Markdown("上传视频文件，系统将自动分析视频内容的真实性")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入控件
                video_input = gr.Video(label="上传视频文件")
                video_title = gr.Textbox(label="视频标题（可选）", placeholder="请输入视频标题")
                use_preprocessing = gr.Checkbox(label="启用预处理（推荐）", value=True)
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 开始分析", variant="primary")
                    export_btn = gr.Button("📥 导出结果")
                
                status_output = gr.Textbox(label="状态", interactive=False)
                download_file = gr.File(label="下载分析结果", visible=False)
        
        with gr.Column(scale=2):
            # 结果展示
            with gr.Tabs():
                with gr.TabItem("转录与帧描述"):
                    transcription_output = gr.Textbox(label="音频转录", lines=5, interactive=False)
                    frame_descriptions_output = gr.Textbox(label="帧描述", lines=10, interactive=False)
                
                with gr.TabItem("分析结果"):
                    consistency_output = gr.Textbox(label="一致性分析", lines=5, interactive=False)
                    ai_detection_output = gr.Textbox(label="AI检测", lines=5, interactive=False)
                    offensive_output = gr.Textbox(label="冒犯性语言检测", lines=5, interactive=False)
                
                with gr.TabItem("事实核查"):
                    fact_checking_output = gr.Textbox(label="事实核查", lines=5, interactive=False)
                    external_evidence_output = gr.Textbox(label="外部证据", lines=5, interactive=False)
                    suspicious_segments_output = gr.Textbox(label="可疑片段定位", lines=5, interactive=False)
                
                with gr.TabItem("最终分析"):
                    final_analysis_output = gr.Textbox(label="综合分析结果", lines=10, interactive=False)
        
        # 事件绑定
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, video_title, use_preprocessing],
            outputs=[
                status_output,
                transcription_output,
                frame_descriptions_output,
                consistency_output,
                ai_detection_output,
                offensive_output,
                fact_checking_output,
                external_evidence_output,
                suspicious_segments_output,
                final_analysis_output
            ]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[download_file, status_output]
        )
        
        # 示例视频（如果有）
        gr.Examples(
            examples=[
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/6793884900228336911.mp4", "真假口罩区别在于中间层，假的一点即燃，真的不起火", True],
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/3xsmc8srtfqe4dc.mp4", "太阳爆发强耀斑对我国长生了影响官方预报未来三天可能会出现地磁暴太阳耀斑爆发", True],
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/BV1nV411W7aY.mp4", "SEVENTEEN\n“厌男”手势\nRabbit每日爆料\n次人绝对是故意的\n所以不要错过全部舞台\n金珉奎你破音了\n稍微\n就是这么自信\n把保护打在公屏上\n等一下吧\n格局要大", True],
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/7038944755274829096.mp4", "买个做保姆挺不错\n😁😁", True],
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/7274921328162016572.mp4", "日本将核污水排海海洋生物会有怎样的变化", True],
                ["/root/siton-tmp/projects/paper/ours/www2026/FakeAgent/examples/7263761519211482405.mp4", "第二集被核污染影响的动物能有多可怕", True],
            ],
            inputs=[video_input, video_title, use_preprocessing]
        )
        # gr.Examples(
        #     examples=[
        #         ["zZB5xaS5SNA.mp4", "Space X’s First Civilian Spacewalk Undertaken By Polaris Dawn Crew", True]
        #     ],
        #     inputs=[video_input, video_title, use_preprocessing],
        #     examples_dir="examples"  # 显式指定
        # )
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)