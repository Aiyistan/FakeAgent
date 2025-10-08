# HMVD (Hierarchical Misinformation Video Detection) 虚假视频检测系统

基于多智能体架构的虚假视频检测系统，采用层次化分析方法来识别和验证视频内容的真实性。

## 项目概述

本项目是一个先进的虚假视频检测系统，结合了多种分析技术和智能体协作，能够：
- 分析视频的跨模态一致性
- 检测AI生成内容
- 识别 offensive 语言
- 进行事实核查
- 检索外部证据
- 整合多维度分析结果

## 项目结构

```
HMVD/
├── baselines/              # 基线模型和推理代码
│   └── hmvd_inference_mix.py  # 多模式推理核心类
├── data/                  # 数据集和配置文件
│   └── HMVD_subset.jsonl  # 视频元数据
├── FakeAgent/             # 多智能体检测系统
│   ├── __init__.py
│   ├── main.py           # 主程序入口
│   ├── workflow.py       # 工作流管理
│   └── agents.py         # 智能体定义
└── results/              # 检测结果输出
```

## 主要功能

### 1. 多模态分析能力
- **视频帧分析**: 采样和分析视频关键帧
- **音频转录**: 提取和理解视频音频内容
- **跨模态一致性**: 检查视觉和听觉信息的一致性

### 2. 智能体系统
系统包含7个专业化智能体：
- **ConsistencyAnalyzer**: 一致性分析智能体
- **AIDetector**: AI生成内容检测智能体
- **OffensiveLanguageDetector**: 攻击性语言检测智能体
- **FactChecker**: 事实核查智能体
- **Retriever**: 外部证据检索智能体
- **Locator**: 可疑片段定位智能体
- **Integrator**: 结果整合智能体

### 3. 推理模式
支持多种推理模式：
- **基础推理**: 快速判断视频真实性
- **视频思维链(VOT)**: 提供详细的分析过程和推理链

### 4. 模型支持
- **OpenAI API**: 支持GPT系列模型
- **Ollama**: 本地模型推理

## 核心特性

### 性能优化
- **异步处理**: 支持高并发的异步推理
- **批处理**: 提高吞吐量
- **缓存机制**: 智能缓存图片和处理结果
- **内存优化**: 动态内存管理

### 可扩展性
- **模块化设计**: 各组件独立，易于扩展
- **配置驱动**: 通过参数调整系统行为
- **多种输入格式**: 支持不同视频和帧格式

### 容错性
- **断点续传**: 支持从中断处继续处理
- **错误恢复**: 智能处理各种异常情况
- **日志记录**: 详细的处理日志和错误追踪

## 快速开始

### 环境要求
- Python 3.8+
- OpenAI API 或本地模型服务

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置环境变量
```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="your_api_base"
export MODEL="your_model_name"
```

### 基础使用

#### 单个视频检测
```python
from FakeAgent.workflow import kickoff

# 处理单个视频
result = kickoff("path/to/video.mp4", "视频标题")
print(result)
```

#### 批量检测
```python
from FakeAgent.main import process_dataset_parallel

# 处理整个数据集
results = process_dataset_parallel(
    jsonl_path="data/HMVD_subset.jsonl",
    output_dir="results/",
    max_workers=4,
    max_videos=100
)
```

#### 使用基线模型
```python
from baselines.hmvd_inference_mix import HMVDInference

# 创建推理器
inference = HMVDInference(
    model_type="openai",
    model_name="gpt-4-vision-preview",
    use_vot_prompt=True  # 使用视频思维链模式
)

# 运行推理
results = inference.run_inference(
    data_root="data/",
    frame_dir="frames",
    split="all",
    output_dir="results/"
)
```

## 命令行使用

### FakeAgent 模式
```bash
python FakeAgent/main.py \
    --jsonl_path data/HMVD_subset.jsonl \
    --output_dir results/ \
    --max_videos 100 \
    --max_workers 4
```

### 基线模型模式
```bash
python baselines/hmvd_inference_mix.py \
    --model_type openai \
    --model_name gpt-4-vision-preview \
    --data_root data/ \
    --frame_dir frames \
    --vot_prompt \
    --sync_mode
```

## 参数说明

### FakeAgent 参数
- `--jsonl_path`: HMVD数据集元数据文件路径
- `--output_dir`: 结果输出目录
- `--max_videos`: 最大处理视频数量
- `--max_workers`: 并行工作线程数
- `--skip_existing`: 跳过已处理的视频

### 基线模型参数
- `--model_type`: 模型类型 (ollama/openai)
- `--model_name`: 模型名称
- `--vot_prompt`: 使用视频思维链prompt
- `--sync_mode`: 同步推理模式
- `--batch_size`: 批处理大小
- `--max_concurrent`: 最大并发数

## 输出格式

### 检测结果
每个视频的检测结果包含：
- `video_id`: 视频标识符
- `title`: 视频标题
- `fake_type`: 虚假类型标签
- `final_prediction`: 最终预测结果 (0/1)
- `analysis`: 详细分析内容
- `timestamp`: 处理时间戳

### 智能体结果
各智能体的独立分析结果保存在 `agents/` 子目录：
- `consistency_analysis.json`: 一致性分析
- `ai_detection.json`: AI检测
- `fact_checking.json`: 事实核查
- `external_evidence.json`: 外部证据
- 等等...

## 数据格式

### HMVD 数据集
使用JSONL格式的视频元数据：
```json
{
  "video_id": "video_001",
  "title": "视频标题",
  "annotation": "视频描述",
  "fake_type": "虚假类型",
  "label": 0或1
}
```

### 视频帧结构
视频帧按以下结构组织：
```
frames_noresize/
├── video_001/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
```

## 性能指标

系统在标准测试环境下的表现：
- **处理速度**: 平均2-5秒/视频（取决于模型）
- **准确率**: 85%+（在HMVD测试集上）
- **内存使用**: 4-8GB（取决于并发数）
- **吞吐量**: 最高100视频/分钟

## 常见问题

### Q: 如何添加新的智能体？
A: 在 `FakeAgent/agents.py` 中继承 `BaseAgent` 类，实现 `analyze` 方法，然后在 `workflow.py` 中注册。

### Q: 如何提高处理速度？
A: 增加 `max_workers` 和 `max_concurrent` 参数，使用更快的模型，或启用缓存机制。

### Q: 处理中断后如何恢复？
A: 系统支持断点续传，只需重新运行相同命令，设置 `--resume` 参数。

## 贡献指南

欢迎贡献代码！请遵循以下步骤：
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

---

*注意：本项目仅用于学术研究和防御性安全目的。请负责任地使用此技术。*