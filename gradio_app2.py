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
            audio_model_dir="/data/yyf/model/SenseVoiceSmall/",
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
    """创建Gradio界面 - 包含清空与自动重置功能"""
    with gr.Blocks(title="FakeAgent 虚假视频检测", theme=gr.themes.Soft()) as demo:
        
        # 定义辅助函数：仅重置输出区域（用于视频变更时）
        def reset_outputs():
            return (
                gr.Markdown("### ⏳ 等待分析..."), # 状态
                "", "", "", "", "", "", "", "",   # 各个Textbox
                gr.Markdown("请分析视频以获取综合报告...", visible=True), # 最终分析
                None  # 下载文件
            )

        # 定义辅助函数：清空所有内容（用于清空按钮）
        def clear_all():
            return (
                None,   # 视频
                "",     # 标题
                True,   # 预处理勾选
                *reset_outputs() # 解包调用上面的重置输出
            )

        gr.Markdown(
            """
            # 🕵️ FakeAgent 虚假视频检测系统
            ### 🤖 多模态视频真实性分析平台
            """
        )
        
        with gr.Row():
            # --- 左侧：控制面板 ---
            with gr.Column(scale=4):
                # 1. 视频输入 (固定高度)
                video_input = gr.Video(
                    label="上传视频文件", 
                    height=450, 
                    width="100%"
                )
                
                with gr.Accordion("⚙️ 高级选项 & 设置", open=False):
                    video_title = gr.Textbox(label="视频标题（辅助分析）", placeholder="请输入视频标题")
                    use_preprocessing = gr.Checkbox(label="启用预处理", value=True)
                
                # 2. 按钮区
                with gr.Row():
                    analyze_btn = gr.Button("🔍 开始分析", variant="primary", scale=2)
                    # 新增：清空按钮
                    clear_btn = gr.Button("🗑️ 清空重置", variant="stop", scale=1)
                
                export_btn = gr.Button("📥 导出结果 JSON")
                
                # 状态显示
                status_output = gr.Markdown("### ⏳ 等待上传...")
                download_file = gr.File(label="下载分析结果", visible=False)

            # --- 右侧：分析报告 ---
            with gr.Column(scale=6):
                gr.Markdown("## 📊 分析报告")
                
                # 综合结果
                with gr.Group():
                    final_analysis_output = gr.Markdown(
                        value="请分析视频以获取综合报告...", 
                        label="综合分析结果"
                    )

                # 详细结果 Tabs
                with gr.Tabs():
                    with gr.TabItem("🔍 详细检测指标"):
                        with gr.Row():
                            with gr.Column():
                                ai_detection_output = gr.Textbox(label="AI 生成检测", lines=3, show_copy_button=True)
                                consistency_output = gr.Textbox(label="音画一致性分析", lines=4, show_copy_button=True)
                            with gr.Column():
                                fact_checking_output = gr.Textbox(label="事实核查", lines=3, show_copy_button=True)
                                offensive_output = gr.Textbox(label="冒犯性语言", lines=3, show_copy_button=True)
                        
                        external_evidence_output = gr.Textbox(label="外部证据搜索", lines=3)
                        suspicious_segments_output = gr.Textbox(label="可疑片段定位", lines=2)

                    with gr.TabItem("📝 原始数据"):
                        transcription_output = gr.Textbox(label="音频转录内容", lines=8, show_copy_button=True)
                        frame_descriptions_output = gr.Textbox(label="关键帧视觉描述", lines=8, show_copy_button=True)

        # ------------------- 事件绑定 -------------------

        # 1. 输出组件列表 (方便复用)
        output_components = [
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

        # 2. 【核心逻辑】开始分析
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, video_title, use_preprocessing],
            outputs=output_components
        )
        
        # 3. 导出按钮
        export_btn.click(
            fn=export_results,
            outputs=[download_file, status_output]
        )

        # 4. 【新功能】监听视频变化 -> 自动重置输出
        # 当用户上传新视频，或点击Example时，自动触发此逻辑
        video_input.change(
            fn=reset_outputs,
            inputs=None,
            outputs=output_components + [download_file] # 同时隐藏下载文件
        )

        # 5. 【新功能】清空按钮 -> 重置所有输入和输出
        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[video_input, video_title, use_preprocessing] + output_components + [download_file]
        )

        # 示例数据
        gr.Examples(
            examples=[
                ["/data/yyf/paper/www2026/FakeAgent/examples/6793884900228336911.mp4", "真假口罩区别在于中间层，假的一点即燃，真的不起火", True],
                ["/data/yyf/paper/www2026/FakeAgent/examples/3xsmc8srtfqe4dc.mp4", "太阳爆发强耀斑对我国长生了影响官方预报未来三天可能会出现地磁暴太阳耀斑爆发", True],
                ["/data/yyf/paper/www2026/FakeAgent/examples/BV1nV411W7aY.mp4", "SEVENTEEN\n“厌男”手势\nRabbit每日爆料\n次人绝对是故意的\n所以不要错过全部舞台\n金珉奎你破音了\n稍微\n就是这么自信\n把保护打在公屏上\n等一下吧\n格局要大", True],
                ["/data/yyf/paper/www2026/FakeAgent/examples/7038944755274829096.mp4", "买个做保姆挺不错\n😁😁", True],
                ["/data/yyf/paper/www2026/FakeAgent/examples/7274921328162016572.mp4", "日本将核污水排海海洋生物会有怎样的变化", True],
                ["/data/yyf/paper/www2026/FakeAgent/examples/7263761519211482405.mp4", "第二集被核污染影响的动物能有多可怕", True],
            ],
            inputs=[video_input, video_title, use_preprocessing]
        )
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7864, share=True)