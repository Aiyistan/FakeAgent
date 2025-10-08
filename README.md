# From Manipulation to Mistrust: Explaining Diverse Micro-Video Misinformation for Robust Debunking in the Wild

**FakeAgent** - A multi-agent system for robust video misinformation detection and explanation.

## 🚀 Quick Start

```bash
cd FakeAgent

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="your_api_base"
```

## 📁 Project Structure

```
FakeAgent/
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
