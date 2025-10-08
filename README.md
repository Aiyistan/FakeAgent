# From Manipulation to Mistrust: Explaining Diverse Micro-Video Misinformation for Robust Debunking in the Wild

**HMVD (Hierarchical Misinformation Video Detection)** - A multi-agent system for robust video misinformation detection and explanation.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Aiyistan/HMVD.git
cd HMVD

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="your_api_base"
```

## 📁 Project Structure

```
HMVD/
├── baselines/          # Baseline models and inference
│   └── hmvd_inference_mix.py
├── FakeAgent/         # Multi-agent detection system
│   ├── main.py        # Main entry point
│   ├── workflow.py    # Workflow management
│   └── agents.py      # Agent definitions
├── data/              # Dataset and metadata
└── requirements.txt   # Dependencies
```

## 🎯 Key Features

### Multi-Agent Architecture
- **Consistency Analyzer**: Cross-modal consistency analysis
- **AI Detector**: AI-generated content detection
- **Fact Checker**: Automated fact verification
- **Retriever**: External evidence retrieval
- **Integrator**: Multi-source result synthesis

### Inference Modes
- **Basic Mode**: Fast misinformation classification
- **Video Thought Chain (VOT)**: Detailed reasoning and explanation

### Model Support
- **OpenAI API**: GPT-4V, GPT-4o, and other vision models
- **Ollama**: Local model inference (Llama 3.2, etc.)

## 💻 Usage

### Single Video Detection
```python
from FakeAgent.workflow import kickoff

result = kickoff("path/to/video.mp4", "Video Title")
```

### Batch Processing
```python
from FakeAgent.main import process_dataset_parallel

results = process_dataset_parallel(
    jsonl_path="data/HMVD_subset.jsonl",
    output_dir="results/",
    max_workers=4
)
```

### Baseline Models
```python
from baselines.hmvd_inference_mix import HMVDInference

inference = HMVDInference(
    model_type="openai",
    model_name="gpt-4-vision-preview",
    use_vot_prompt=True
)

results = inference.run_inference(
    data_root="data/",
    frame_dir="frames",
    output_dir="results/"
)
```

## ⚡ Performance

- **Accuracy**: 85%+ on HMVD test set
- **Speed**: 2-5 seconds per video (model-dependent)
- **Throughput**: Up to 100 videos/minute
- **Memory**: 4-8GB (depends on concurrency)

## 🔧 Command Line

### FakeAgent Mode
```bash
python FakeAgent/main.py \
    --jsonl_path data/HMVD_subset.jsonl \
    --output_dir results/ \
    --max_workers 4
```

### Baseline Mode
```bash
python baselines/hmvd_inference_mix.py \
    --model_type openai \
    --model_name gpt-4-vision-preview \
    --vot_prompt \
    --data_root data/ \
    --frame_dir frames
```

## 📊 Output Format

```json
{
  "video_id": "video_001",
  "title": "Video Title",
  "fake_type": "misinformation_type",
  "final_prediction": "0 or 1",
  "analysis": "Detailed explanation...",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This project is designed for academic research and defensive security purposes only. Please use responsibly.

🌐 **Project Page**: https://github.com/Aiyistan/HMVD