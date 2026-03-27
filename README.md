# FakeAgent

**From Manipulation to Mistrust: Explaining Diverse Micro-Video Misinformation for Robust Debunking in the Wild**

[[Paper](https://arxiv.org/pdf/2603.25423)]

A multi-agent system for robust video misinformation detection and explanation.

## 🚀 Quick Start

```bash
cd FakeAgent

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="your_api_base"
export SERPER_API_KEYS="your_serper_api_key"
```

## 📁 Project Structure

```
FakeAgent/
├── agents/           # Multi-agent definitions
│   ├── consistency_analyzer.py
│   ├── ai_detector.py
│   ├── fact_checker.py
│   ├── retriever.py
│   └── integrator.py
├── workflows/        # Detection workflow
│   └── detector.py
├── utils/            # Utility modules
│   ├── audio_extractor.py
│   ├── video_frame_extractor.py
│   └── frame_caption_openai.py
├── main.py           # Main entry point
├── gradio_app.py     # Gradio web interface
└── requirements.txt  # Dependencies
```

## 🎯 Key Features

**Multi-Agent Architecture**
- **Consistency Analyzer**: Cross-modal consistency analysis (video-audio-title)
- **AI Detector**: AI-generated content detection
- **Offensive Language Detector**: Harmful content detection
- **Fact Checker**: Automated fact verification
- **Retriever**: External evidence retrieval via web search
- **Locator**: Suspicious segment localization
- **Integrator**: Multi-source result synthesis

**Inference Modes**
- **Basic Mode**: Fast misinformation classification
- **Full Pipeline**: Detailed reasoning with evidence retrieval and explanation

## 📖 Usage

```bash
# Process single video
python main.py --single_video /path/to/video.mp4 --video_title "Video Title"

# Batch processing
python main.py --jsonl_path /path/to/dataset.jsonl \
               --output_dir ./results \
               --max_videos 10 \
               --max_workers 4 \
               --use_preprocessing

# Launch Gradio web interface
python gradio_app.py
```

## 📩 Dataset Access

If you need access to the dataset used in this project, please contact via email: **yangyf001@stu.xjtu.edu.cn**
