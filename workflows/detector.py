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
from pathlib import Path

try:
    # 尝试使用相对导入（当作为模块导入时）
    from ..agents import (
        ConsistencyAnalyzer,
        AIDetector,
        OffensiveLanguageDetector,
        FactChecker,
        Retriever,
        Locator,
        Integrator,
    )
    from ..utils.video_frame_extractor import VideoFrameExtractor
    from ..utils.frame_caption_openai import VisionInferencer
    from ..utils.audio_extractor import AudioExtractor
except ImportError:
    # 如果相对导入失败，使用绝对导入（当直接运行时）
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from agents import (
        ConsistencyAnalyzer,
        AIDetector,
        OffensiveLanguageDetector,
        FactChecker,
        Retriever,
        Locator,
        Integrator,
    )
    from utils.video_frame_extractor import VideoFrameExtractor
    from utils.frame_caption_openai import VisionInferencer
    from utils.audio_extractor import AudioExtractor


class FakeVideoDetectorWorkflow:
    """虚假视频检测工作流（优化版）"""
    def __init__(self, max_workers=8, 
                 video_output_dir=None, 
                 frame_caption_model_path=None,
                 audio_model_dir=None,
                 audio_device=0,
                 openai_api_base=None,
                 openai_api_key=None,
                 openai_model="gpt-4-vision-preview"):
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
        
        # 预处理工具初始化
        self.video_output_dir = video_output_dir
        self.frame_caption_model_path = frame_caption_model_path
        self.audio_model_dir = audio_model_dir
        self.audio_device = audio_device
        
        # OpenAI API 相关参数
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        
        # 延迟初始化，只在需要时创建
        self._frame_extractor = None
        self._frame_caption_inferencer = None
        self._audio_extractor = None

    @lru_cache(maxsize=32)
    def _load_json_file(self, file_path: str) -> Dict:
        """缓存JSON文件加载结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @property
    def frame_extractor(self):
        """延迟初始化视频帧提取器"""
        if self._frame_extractor is None:
            self._frame_extractor = VideoFrameExtractor(
                video_dir=os.path.dirname(self.state.get('video_path', '')),
                output_dir=os.path.join(self.video_output_dir, 'frames'),
                num_frames=8
            )
        return self._frame_extractor
    
    @property
    def frame_caption_inferencer(self):
        """延迟初始化帧描述推理器"""
        if self._frame_caption_inferencer is None and self.openai_api_base and self.openai_api_key:
            self._frame_caption_inferencer = VisionInferencer(
                api_base=self.openai_api_base,
                api_key=self.openai_api_key,
                model=self.openai_model,
                timeout=60
            )
        return self._frame_caption_inferencer
    
    @property
    def audio_extractor(self):
        """延迟初始化音频提取器"""
        if self._audio_extractor is None and self.audio_model_dir:
            self._audio_extractor = AudioExtractor(
                model_dir=self.audio_model_dir,
                device=self.audio_device,
                video_base_dir=os.path.dirname(self.state.get('video_path', '')),
                audio_output_dir=os.path.join(self.video_output_dir, "audio")
            )
        return self._audio_extractor
    
    def preprocess_video(self, video_path: str) -> Dict[str, Any]:
        """
        预处理视频：提取帧、生成帧描述、提取和转录音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含预处理结果的字典
        """
        video_id = Path(video_path).stem
        result = {
            'video_id': video_id,
            'frames': [],
            'frame_times': [],
            'frame_descriptions': [],
            'transcript': '无'
        }
        
        # 定义两个JSON文件路径
        frame_caption_file = os.path.join(self.video_output_dir, f"{video_id}_frame_caption.json")
        transcript_file = os.path.join(self.video_output_dir, f"{video_id}_transcript.json")
        
        # 确保输出目录存在
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # 检查是否已经处理过
        if os.path.exists(frame_caption_file) and os.path.exists(transcript_file):
            print(f"发现已处理的视频 {video_id}，直接加载结果...")
            try:
                # 加载帧描述结果
                with open(frame_caption_file, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    result['frame_descriptions'] = frame_data.get('frame_descriptions', [])
                    result['frame_times'] = frame_data.get('frame_times', [])
                    result['frames'] = frame_data.get('frames', [])
                
                # 加载转录结果
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    result['transcript'] = transcript_data.get('transcript', '无')
                
                print(f"成功加载已有处理结果")
                return result
            except Exception as e:
                print(f"加载已有处理结果失败: {e}，将重新处理...")
        
        try:
            # 1. 提取视频帧
            print(f"开始提取视频帧: {video_id}")
            frames, frame_times = self.frame_extractor.extract_frames_uniform(
                video_path, video_id, quality_aware=True
            )
            result['frames'] = frames
            result['frame_times'] = frame_times
            print(f"成功提取 {len(frames)} 帧")
            
            # 2. 生成帧描述（如果有模型）
            if self.frame_caption_inferencer:
                print(f"开始生成帧描述: {video_id}")
                try:
                    frame_descriptions = self.frame_caption_inferencer.batch_infer(frames, prompt="请用100字左右描述这个视频帧中的新闻事件或短视频传达的内容。")
                    result['frame_descriptions'] = frame_descriptions
                    print(f"成功生成 {len(frame_descriptions)} 个帧描述")
                except Exception as e:
                    print(f"生成帧描述失败: {e}")
                    result['frame_descriptions'] = ["无画面描述"] * len(frames)
            else:
                print("跳过帧描述生成（未提供OpenAI API配置）")
                result['frame_descriptions'] = ["无画面描述"] * len(frames)
            
            # 3. 提取和转录音频（如果有模型）
            if self.audio_extractor:
                print(f"开始提取和转录音频: {video_id}")
                try:
                    # 提取音频
                    audio_path = self.audio_extractor.extract_audio_from_video(video_path)
                    
                    # 转录音频
                    transcript = self.audio_extractor.transcribe_audio(str(audio_path))
                    result['transcript'] = transcript
                    print(f"成功转录音频: {transcript[:50]}...")
                except Exception as e:
                    print(f"音频提取或转录失败: {e}")
                    result['transcript'] = '无'
            else:
                print("跳过音频提取和转录（未提供模型路径）")
                result['transcript'] = '无'
            
            # 保存处理结果到两个JSON文件
            try:
                # 保存帧描述和相关信息
                frame_data = {
                    'video_id': video_id,
                    'frames': result['frames'],
                    'frame_times': result['frame_times'],
                    'frame_descriptions': result['frame_descriptions']
                }
                with open(frame_caption_file, 'w', encoding='utf-8') as f:
                    json.dump(frame_data, f, ensure_ascii=False, indent=2)
                print(f"帧描述结果已保存到 {frame_caption_file}")
                
                # 保存转录结果
                transcript_data = {
                    'video_id': video_id,
                    'transcript': result['transcript']
                }
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                print(f"转录结果已保存到 {transcript_file}")
            except Exception as e:
                print(f"保存处理结果失败: {e}")
                
        except Exception as e:
            print(f"视频预处理失败: {e}")
            # 返回默认值，确保后续流程可以继续
            result['frames'] = []
            result['frame_times'] = []
            result['frame_descriptions'] = ["无画面描述"]
            result['transcript'] = '无'
            
        return result

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

            # 首先检查是否有预处理生成的转录文件
            if self.video_output_dir:
                transcript_file = os.path.join(self.video_output_dir, f"{video_id}_transcript.json")
                if os.path.exists(transcript_file):
                    data = await self._async_load_json_file(transcript_file)
                    transcript = data.get('transcript', '无')
                    if transcript and transcript != '无':
                        self.state["transcription"] = transcript
                        self.cache[cache_key] = transcript  # 缓存结果
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

            # 首先检查是否有预处理生成的帧描述文件
            if self.video_output_dir:
                frame_caption_file = os.path.join(self.video_output_dir, f"{video_id}_frame_caption.json")
                if os.path.exists(frame_caption_file):
                    data = await self._async_load_json_file(frame_caption_file)
                    frames = data.get('frames', [])
                    frame_timestamps = data.get('frame_times', [])
                    frame_descriptions = data.get('frame_descriptions', [])
                    
                    if frame_descriptions:
                        # 添加时间戳信息
                        frame_descriptions_with_timestamps = [
                            f"[timestamp={timestamp}s][frame_description]: {description}"
                            for description, timestamp in zip(frame_descriptions, frame_timestamps)
                        ]
                        self.state["frame_descriptions"] = frame_descriptions_with_timestamps
                        self.cache[cache_key] = frame_descriptions_with_timestamps  # 缓存结果
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
        
        # 获取视频ID
        video_id = self.state['video_path'].split('/')[-1].split('.')[0]
        
        # 定义分析结果保存路径
        analysis_results_file = os.path.join(self.video_output_dir, f"{video_id}_analysis_results.json")
        
        # 确保输出目录存在
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # 检查是否已有分析结果
        if os.path.exists(analysis_results_file):
            print(f"发现已存在的分析结果文件，直接加载: {analysis_results_file}")
            try:
                with open(analysis_results_file, 'r', encoding='utf-8') as f:
                    analysis_results = json.load(f)
                
                # 加载分析结果到状态中
                self.state['consistency_analysis'] = analysis_results.get('consistency_analysis', '')
                self.state['ai_detection'] = analysis_results.get('ai_detection', '')
                self.state['offensive_language_detection'] = analysis_results.get('offensive_language_detection', '')
                self.state['fact_checking'] = analysis_results.get('fact_checking', '')
                
                print("成功加载已有分析结果")
            except Exception as e:
                print(f"加载已有分析结果失败: {e}，将重新进行分析...")
        else:
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

            # 保存分析结果
            try:
                analysis_results = {
                    'video_id': video_id,
                    'consistency_analysis': self.state.get('consistency_analysis', ''),
                    'ai_detection': self.state.get('ai_detection', ''),
                    'offensive_language_detection': self.state.get('offensive_language_detection', ''),
                    'fact_checking': self.state.get('fact_checking', '')
                }
                
                with open(analysis_results_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, ensure_ascii=False, indent=2)
                
                print(f"分析结果已保存到: {analysis_results_file}")
            except Exception as e:
                print(f"保存分析结果失败: {e}")

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

            if fact_checking["need_external_evidence"] == '否':
                self.state["need_external_evidence"] = False
            else:
                self.state["need_external_evidence"] = True
        except Exception as e:
            print(f"解析事实检查结果失败: {e}")
            self.state["need_external_evidence"] = False
        
        
    def run_evidence_retrieval(self):
        """运行外部证据检索和可疑片段定位"""
        # 获取视频ID
        video_id = self.state['video_path'].split('/')[-1].split('.')[0]
        
        # 定义证据检索结果保存路径
        evidence_results_file = os.path.join(self.video_output_dir, f"{video_id}_evidence_results.json")
        
        # 确保输出目录存在
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # 检查是否已有证据检索结果
        if os.path.exists(evidence_results_file):
            print(f"发现已存在的证据检索结果文件，直接加载: {evidence_results_file}")
            try:
                with open(evidence_results_file, 'r', encoding='utf-8') as f:
                    evidence_results = json.load(f)
                
                # 加载证据检索结果到状态中
                self.state['external_evidence'] = evidence_results.get('external_evidence', '未检索外部证据')
                self.state['suspicious_segments'] = evidence_results.get('suspicious_segments', '未定位可疑片段')
                
                print("成功加载已有证据检索结果")
                return
            except Exception as e:
                print(f"加载已有证据检索结果失败: {e}，将重新进行检索...")
        
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
            
        # 保存证据检索结果
        try:
            evidence_results = {
                'video_id': video_id,
                'external_evidence': self.state.get('external_evidence', '未检索外部证据'),
                'suspicious_segments': self.state.get('suspicious_segments', '未定位可疑片段')
            }
            
            with open(evidence_results_file, 'w', encoding='utf-8') as f:
                json.dump(evidence_results, f, ensure_ascii=False, indent=2)
            
            print(f"证据检索结果已保存到: {evidence_results_file}")
        except Exception as e:
            print(f"保存证据检索结果失败: {e}")

    def run_integrator(self):
        """运行整合器"""
        # 获取视频ID
        video_id = self.state['video_path'].split('/')[-1].split('.')[0]
        
        # 定义整合结果保存路径
        integration_results_file = os.path.join(self.video_output_dir, f"{video_id}_integration_results.json")
        
        # 确保输出目录存在
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # 检查是否已有整合结果
        if os.path.exists(integration_results_file):
            print(f"发现已存在的整合结果文件，直接加载: {integration_results_file}")
            try:
                with open(integration_results_file, 'r', encoding='utf-8') as f:
                    integration_results = json.load(f)
                
                # 加载整合结果到状态中
                self.state['analysis'] = integration_results.get('analysis', '')
                
                print("成功加载已有整合结果")
                return
            except Exception as e:
                print(f"加载已有整合结果失败: {e}，将重新进行整合...")
        
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
        
        # 保存整合结果
        try:
            integration_results = {
                'video_id': video_id,
                'analysis': self.state.get('analysis', '')
            }
            
            with open(integration_results_file, 'w', encoding='utf-8') as f:
                json.dump(integration_results, f, ensure_ascii=False, indent=2)
            
            print(f"整合结果已保存到: {integration_results_file}")
        except Exception as e:
            print(f"保存整合结果失败: {e}")

    async def run_workflow_async(self, video_path: str, video_title: str = "", transcript_json: str="", frame_caption_json: str="", 
                              use_preprocessing: bool = False):
        """异步运行完整的工作流"""
        try:
            # if not os.path.exists(video_path):
            #     raise FileNotFoundError(f"视频文件不存在: {video_path}")

            # print(f"开始检测视频: {video_path}")
            # print(f"视频标题: {video_title}")

            # 初始化状态
            self.initialize_state(video_path, video_title, transcript_json, frame_caption_json)
            
            # 如果启用预处理，则执行预处理
            if use_preprocessing:
                print("开始视频预处理...")
                # 执行预处理（内部会检查是否已有处理结果）
                preprocess_result = await asyncio.get_event_loop().run_in_executor(
                    None, self.preprocess_video, video_path
                )
                
                # 更新状态中的转录和帧描述
                if preprocess_result['transcript'] and preprocess_result['transcript'] != '无':
                    self.state["transcription"] = preprocess_result['transcript']
                
                if preprocess_result['frame_descriptions']:
                    # 添加时间戳信息
                    frame_descriptions_with_timestamps = [
                        f"[timestamp={timestamp}s][frame_description]: {description}"
                        for description, timestamp in zip(preprocess_result['frame_descriptions'], preprocess_result['frame_times'])
                    ]
                    self.state["frame_descriptions"] = frame_descriptions_with_timestamps
                
                print("视频预处理完成")
            else:
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
            # # 随机休眠8-15秒
            # await asyncio.sleep(random.uniform(8, 15))

            print("检测流程完成")
            return self.state
        except Exception as e:
            print(f"工作流执行失败: {e}")
            self.state["error"] = str(e)
            return self.state

    def run_workflow(self, video_path: str, video_title: str = "", transcript_json: str="", frame_caption_json: str="", 
                  use_preprocessing: bool = False):
        """运行完整的工作流（同步接口）"""
        return asyncio.run(self.run_workflow_async(video_path, video_title, transcript_json, frame_caption_json, use_preprocessing))

    def batch_process(self, video_list: List[Dict[str, str]], use_preprocessing: bool = False):
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
                    video.get("frame_caption_json", ""),
                    use_preprocessing
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


def kickoff(video_path: str, video_title: str = "", transcript_json: str = "", frame_caption_json: str="", 
             use_preprocessing: bool = False, 
             video_output_dir=None, 
             frame_caption_model_path=None,
             audio_model_dir=None,
             audio_device=0,
             openai_api_base=os.getenv('MULTIMODAL_OPENAI_API_BASE'),
             openai_api_key=os.getenv('MULTIMODAL_OPENAI_API_KEY'),
             openai_model=os.getenv('MULTIMODAL_OPENAI_MODEL')):
    """启动检测流程"""
    # 预处理: 检查是否有video_path的音频转录和视频抽帧caption,如果没有先生成
    video_output_dir =  video_output_dir or "./tmp/video_preprocess"
    video_output_dir = os.path.join(video_output_dir, os.path.basename(video_path).split(".")[0])
    workflow = FakeVideoDetectorWorkflow(
        video_output_dir=video_output_dir,
        frame_caption_model_path=frame_caption_model_path,
        audio_model_dir=audio_model_dir,
        audio_device=audio_device,
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        openai_model=openai_model
    )
    return workflow.run_workflow(video_path, video_title, transcript_json=transcript_json, 
                               frame_caption_json=frame_caption_json, use_preprocessing=use_preprocessing)


def batch_kickoff(video_list: List[Dict[str, str]], use_preprocessing: bool = False,
                  video_output_dir=None, 
                  frame_caption_model_path=None,
                  audio_model_dir=None,
                  audio_device=0,
                  openai_api_base=None,
                  openai_api_key=None,
                  openai_model="gpt-4-vision-preview"):
    """批量启动检测流程"""
    workflow = FakeVideoDetectorWorkflow(
        video_output_dir=video_output_dir,
        frame_caption_model_path=frame_caption_model_path,
        audio_model_dir=audio_model_dir,
        audio_device=audio_device,
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        openai_model=openai_model
    )
    return workflow.batch_process(video_list, use_preprocessing)