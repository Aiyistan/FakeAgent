import os
import json
from pyexpat import features
import random
import sys
import concurrent.futures
import asyncio
import aiofiles
from typing import Dict, List, Any, Optional
from functools import lru_cache

from ..agents import (
    ConsistencyAnalyzer,
    AIDetector,
    OffensiveLanguageDetector,
    FactChecker,
    Retriever,
    Locator,
    Integrator,
)


class FakeVideoDetectorWorkflow:
    """虚假视频检测工作流（优化版）"""
    def __init__(self, max_workers=8):
        self.state: Dict[str, Any] = {}
        self.agents = {
            'consistency_analyzer': ConsistencyAnalyzer(),
            'ai_detector': AIDetector(),
            'offensive_language_detector': OffensiveLanguageDetector(),
            'fact_checker': FactChecker(),
            'retriever': Retriever(),
            'locator': Locator(),
            'integrator': Integrator(),
            # 'answer': Answer()
        }
        self.max_workers = max_workers  # 最大工作线程数
        self.cache = {}  # 简单缓存机制

    @lru_cache(maxsize=32)
    def _load_json_file(self, file_path: str) -> Dict:
        """缓存JSON文件加载结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _async_load_json_file(self, file_path: str) -> Dict:
        """异步加载JSON文件"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)

    def initialize_state(self, video_path: str, video_title: str = "", transcript_json: str = "", frame_caption_json: str = ""):
        """初始化工作流状态"""
        print("初始化工作流状态...")
        self.state = {
            'video_path': video_path,
            'video_title': video_title,
            'sample_frames': 8,
            'transcription': None,
            'transcript_json': transcript_json,
            'frame_caption_json': frame_caption_json,
            'frame_descriptions': None,
            'consistency_analysis': None,
            'ai_detection': None,
            'offensive_language_detection': None,
            'fact_checking': None,
            'need_external_evidence': False,
            'external_evidence': None,
            'suspicious_segments': None,
            'analysis': None,
            'final_report': None
        }
        print("工作流状态初始化完成")

    async def transcribe_audio(self):
        """异步转录音频内容"""
        try:
            video_id = self.state['video_path'].split('/')[-1].split('.')[0]
            transcript = "无"
            transcript_json = self.state['transcript_json']

            # 检查缓存
            cache_key = f"transcript_{video_id}"
            if cache_key in self.cache:
                self.state["transcription"] = self.cache[cache_key]
                return

            # 检查音频处理日志文件是否存在
            if transcript_json and os.path.exists(transcript_json):
                data = await self._async_load_json_file(transcript_json)
                for item in data:
                    if video_id == item["video_id"]:
                        transcript = item["transcript"]
                        break

            if transcript == "" or transcript is None:
                transcript = "无"

            # print(f"音频转录: {transcript}")
            self.state["transcription"] = transcript
            self.cache[cache_key] = transcript  # 缓存结果
        except Exception as e:
            print(f"转录处理出错: {e}")
            self.state["transcription"] = "无"

    async def describe_frames(self):
        """异步描述视频画面"""
        try:
            video_id = self.state['video_path'].split('/')[-1].split('.')[0]
            
            frames = []
            frame_timestamps = []

            # 检查缓存
            cache_key = f"frames_{video_id}"
            if cache_key in self.cache:
                self.state["frame_descriptions"] = self.cache[cache_key]
                return

            # 检查处理日志文件是否存在
            processing_log_path = self.state['frame_caption_json']
            if processing_log_path and os.path.exists(processing_log_path):
                data = await self._async_load_json_file(processing_log_path)
                for item in data:
                    if item["video_id"] == video_id:
                        frames = item["frames"]
                        frame_timestamps = item["frame_times"]
                        break

            if not frames:
                print(f"未找到视频 {video_id} 的帧数据")
                self.state["frame_descriptions"] = ["无画面描述"]
                return

            # 检查是否已有推理结果
            frame_descriptions = []
            test_json_path = self.state['frame_caption_json']
            if os.path.exists(test_json_path):
                data = await self._async_load_json_file(test_json_path)
                for item in data:
                    if item["video_id"] == video_id:
                        frame_descriptions = item["frame_descriptions"]
                        break

            if not frame_descriptions:
                print(f"未找到视频 {video_id} 的画面描述")
                self.state["frame_descriptions"] = ["无画面描述"]
                return

            # 添加时间戳信息
            frame_descriptions_with_timestamps = [
                f"[timestamp={timestamp}s][frame_description]: {description}"
                for description, timestamp in zip(frame_descriptions, frame_timestamps)
            ]
            # print(f"视频画面描述: {frame_descriptions_with_timestamps}")
            self.state["frame_descriptions"] = frame_descriptions_with_timestamps
            self.cache[cache_key] = frame_descriptions_with_timestamps  # 缓存结果
        except Exception as e:
            print(f"画面描述处理出错: {e}")
            self.state["frame_descriptions"] = ["无画面描述"]

    def run_analysis_agents(self):
        """运行分析代理（并行执行）"""
        print("开始运行分析代理...")
        analysis_inputs = {
            'video_title': self.state.get("video_title", "无"),
            'video_transcription': self.state["transcription"],
            'frame_descriptions': self.state["frame_descriptions"],
        }

        # 定义要并行运行的代理
        agents_to_run = [
            ('consistency_analysis', 'consistency_analyzer'),
            ('ai_detection', 'ai_detector'),
            ('offensive_language_detection', 'offensive_language_detector'),
            ('fact_checking', 'fact_checker')
        ]

        # 并行执行分析代理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.agents[agent].analyze, analysis_inputs): key
                for key, agent in agents_to_run
            }

            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    self.state[key] = future.result()
                    print(f"{key} 分析完成")
                except Exception as e:
                    print(f"{key} 分析失败: {e}")
                    self.state[key] = f"分析失败: {str(e)}"

        # 解析fact_checker结果判断是否需要外部证据
        try:
            fact_checking = json.loads(self.state["fact_checking"])
            confidence_score = fact_checking['confidence_score']

            if float(confidence_score.strip('%')) >= 80.0:
                confidence = '高'
            elif float(confidence_score.strip('%')) >= 60.0:
                confidence = '中'
            else:
                confidence = '低'

            if (fact_checking["need_external_evidence"] == '否' or
                (fact_checking['is_real'] in ['是', '否'] and confidence in ['高', '中'])):
                self.state["need_external_evidence"] = False
            else:
                self.state["need_external_evidence"] = True
        except Exception as e:
            print(f"解析事实检查结果失败: {e}")
            self.state["need_external_evidence"] = False

    def run_evidence_retrieval(self):
        """运行外部证据检索和可疑片段定位"""
        # 提取关键词和核心观点
        fact_checking_str = self.state.get("fact_checking", "")
        try:
            fact_checking = json.loads(fact_checking_str)
            # keywords = fact_checking['external_evidence_keywords']
            core_ideas = fact_checking['external_evidence_core_ideas']
            # print(f"核心观点: {core_ideas}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析fact_checking失败: {e}")
            core_ideas = ""
            # keywords = ""

        # 准备定位代理输入（无论是否需要外部证据都要执行定位）
        locating_inputs = {
            'video_content': f"标题：{self.state.get('video_title', '')}\n转录：{self.state['transcription']}\n画面描述：{' '.join(self.state['frame_descriptions'])}",
            'analysis_results': fact_checking_str,  # 传递原始字符串，代理内部会解析
            'external_evidence': ""  # 初始为空，如果有外部证据会更新
        }

        # 如果需要外部证据，则执行检索
        if self.state.get("need_external_evidence", False):
            print("开始外部证据检索...")
            
            retrieval_inputs = {
                # 'keywords': keywords,
                "video_title": self.state.get("video_title", "无"),
                'core_ideas': core_ideas,
            }

            # 运行检索代理
            try:
                self.state["external_evidence"] = self.agents['retriever'].analyze(retrieval_inputs)
                print("外部证据检索结果:")
                self.state["external_evidence"] = self.state["external_evidence"].split('</think>')[-1].strip()
                # print(self.state["external_evidence"])
                
                # 更新定位输入中的外部证据
                locating_inputs['external_evidence'] = self.state.get("external_evidence", "")
            except Exception as e:
                print(f"外部证据检索失败: {e}")
                self.state["external_evidence"] = "检索失败"
        else:
            print("跳过外部证据检索")
            self.state["external_evidence"] = "未检索外部证据"

        # 无论是否检索到证据，都执行定位
        print("开始可疑片段定位...")
        try:
            self.state["suspicious_segments"] = self.agents['locator'].analyze(locating_inputs)
            # print("虚假内容定位结果:")
            # print(self.state["suspicious_segments"])
        except Exception as e:
            print(f"可疑片段定位失败: {e}")
            self.state["suspicious_segments"] = "定位失败"

    def run_integrator(self):
        """运行整合器"""
        print("开始运行整合器...")
        integration_inputs = {
            'video_title': self.state.get("video_title", "无"),
            'consistency_analysis': self.state.get("consistency_analysis", ""),
            'ai_detection': self.state.get("ai_detection", ""),
            'offensive_language_detection': self.state.get("offensive_language_detection", ""),
            'fact_checking': self.state.get("fact_checking", ""),
            'external_evidence': self.state.get("external_evidence", "未检索外部证据"),
            'suspicious_segments': self.state.get("suspicious_segments", "未定位可疑片段"),
        }
        # print(f"整合器输入: {integration_inputs}")
        self.state["analysis"] = self.agents['integrator'].analyze(integration_inputs)
        # print("整合分析结果:")
        # print(self.state["analysis"])

    async def run_workflow_async(self, video_path: str, video_title: str = "", transcript_json: str="", frame_caption_json: str=""):
        """异步运行完整的工作流"""
        try:
            # if not os.path.exists(video_path):
            #     raise FileNotFoundError(f"视频文件不存在: {video_path}")

            # print(f"开始检测视频: {video_path}")
            # print(f"视频标题: {video_title}")

            # 初始化状态
            self.initialize_state(video_path, video_title, transcript_json, frame_caption_json)

            # 并行执行转录和帧描述
            await asyncio.gather(
                self.transcribe_audio(),
                self.describe_frames()
            )

            # 并行执行分析代理
            self.run_analysis_agents()

            # 执行后续步骤
            self.run_evidence_retrieval()
            self.run_integrator()

            print("检测流程完成")
            return self.state
        except Exception as e:
            print(f"工作流执行失败: {e}")
            self.state["error"] = str(e)
            return self.state

    def run_workflow(self, video_path: str, video_title: str = "", transcript_json: str="", frame_caption_json: str=""):
        """运行完整的工作流（同步接口）"""
        return asyncio.run(self.run_workflow_async(video_path, video_title, transcript_json, frame_caption_json))

    def batch_process(self, video_list: List[Dict[str, str]]):
        """批量处理多个视频"""
        print(f"开始批量处理 {len(video_list)} 个视频...")

        # 使用线程池并行处理多个视频
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_workflow,
                    video['path'],
                    video.get('title', ''),
                    video.get("transcript_json", ""),
                    video.get("frame_caption_json", "")
                ): video['path']
                for video in video_list
            }

            results = {}
            for future in concurrent.futures.as_completed(futures):
                video_path = futures[future]
                try:
                    results[video_path] = future.result()
                    print(f"视频 {video_path} 处理完成")
                except Exception as e:
                    print(f"视频 {video_path} 处理失败: {e}")
                    results[video_path] = {"error": str(e)}

        print("批量处理完成")
        return results


def kickoff(video_path: str, video_title: str = "", transcript_json: str = "", frame_caption_json: str=""):
    """启动检测流程"""
    # 预处理: 检查是否有video_path的音频转录和视频抽帧caption,如果没有先生成
    
    workflow = FakeVideoDetectorWorkflow()
    return workflow.run_workflow(video_path, video_title, transcript_json=transcript_json, frame_caption_json=frame_caption_json)


def batch_kickoff(video_list: List[Dict[str, str]]):
    """批量启动检测流程"""
    workflow = FakeVideoDetectorWorkflow()
    return workflow.batch_process(video_list)