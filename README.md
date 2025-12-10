# D&D Dungeon Master Copilot
## Fine-Tuning Domain-Specific Embeddings for RAG

**Course:** BU CS 506 - Final Project  
**Team:** Garry Kuwanto  
**Final Report:** December 10, 2024

---

## 📹 Final Presentation Video
[![Watch Final Presentation](https://img.youtube.com/vi/1FpNQqdg_r8/hqdefault.jpg)](https://youtu.be/1FpNQqdg_r8)

## 📹 Midterm Presentation Video
[![Watch Midterm Presentation](https://img.youtube.com/vi/OPkHkA7z7tw/hqdefault.jpg)](https://youtu.be/OPkHkA7z7tw)

---

## 🚀 How to Build and Run (Reproduce Results)

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for LFM2 inference)
- ~10GB disk space for models and data

### Step 1: Install Dependencies
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dnd-dm-copilot.git
cd dnd-dm-copilot

# Install all dependencies (includes llama-cpp-python with CUDA support)
make install
```

### Step 2: Set Up Environment
```bash
# Create .env file from template
make dev-setup

# Edit .env with your API keys (required for evaluation)
# - HF_TOKEN: Hugging Face token (for downloading models/datasets)
# - DEEPSEEK_API_KEY: DeepSeek API key (for LLM-as-a-Judge evaluation)
```

### Step 3: Download Pre-trained Models
```bash
# Download the fine-tuned embedding model from Hugging Face
make download-sbert

# Download LFM2 model for answer generation (required for demo)
make download-lfm2

# Or download both at once
make download-models
```


### Step 4: Run the Demo
```bash
# Start the FastAPI server with demo UI
make run-api

# Visit http://localhost:8000/demo/ in your browser
```

### Quick Start (All-in-One)
```bash
make install
make dev-setup
make download-models
make run-api
```

> **Note:** `download-lfm2` downloads ~1GB model file. This is required for the demo to generate answers.

---

## 🧪 How to Test

### Run All Tests
```bash
make test
```

### Run Specific Test Suites
```bash
make test-api      # API endpoint tests
make test-eval     # Evaluation pipeline tests
make test-data     # Data processing tests
make test-model    # Model tests
```

### Run Tests with Coverage
```bash
make test
# Coverage report available at htmlcov/index.html
```

---

## 🖥️ Environment Support

| Component | Supported |
|-----------|-----------|
| **OS** | Linux (tested on Ubuntu 22.04), macOS, Windows (WSL2) |
| **Python** | 3.11+ |
| **GPU** | NVIDIA CUDA 12.4+ (for LFM2 inference) |
| **CPU-only** | Yes (slower inference) |

---

## 📊 Final Results

### Training Distribution (D&D 3.5e) - Retrieval Metrics

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| **Accuracy@1** | 34.8% | **68.2%** | **+96%** |
| **Accuracy@3** | 50.0% | **78.2%** | **+56%** |
| **Accuracy@5** | 55.5% | **82.6%** | **+49%** |
| **MRR@10** | 43.5% | **74.9%** | **+72%** |

### Test Distribution (D&D 5e) - End-to-End Answer Quality

| System | Correct Answers | Accuracy |
|--------|-----------------|----------|
| **Baseline (No RAG)** | 22/500 | **4.4%** |
| **RAG Pipeline (Fine-tuned)** | 206/500 | **41.2%** |

### **+836% Improvement in Answer Quality!**

Our fine-tuned RAG system produces **9x more correct answers** than the baseline model without RAG.

### Visual Evidence: Query-Passage Clustering

![Query-Passage Groups](visualizations/query_passage_groups.png)

- **Baseline Model (Left):** Query-passage pairs scattered with long connecting lines
- **Fine-tuned Model (Right):** Query-passage pairs cluster closer together with shorter lines

---

## 🔬 Addressing Midterm Feedback

> "I question whether these metric gains are truly meaningful... suggest employing additional methods to validate the effectiveness of your accuracy improvements."

### Our Response: Comprehensive Evaluation Pipeline

1. **Different Distribution** - Trained on D&D 3.5e, tested on D&D 5e
2. **LLM-as-a-Judge** - DeepSeek evaluates actual answer quality (not just retrieval)
3. **End-to-End Evaluation** - Full RAG pipeline (retrieval + generation)
4. **Baseline Comparison** - Compare against model without RAG

### Evaluation Pipeline
```
500 5e Passages → Generate Q&A → Run RAG Pipeline → LLM Judge → Metrics
```

This rigorous evaluation demonstrates that our improvements are **meaningful and generalizable**.

---

## 🏗️ System Architecture

```
User Query → Fine-tuned MiniLM → FAISS Search → Top-k Passages → LFM2-1.2B-RAG → Answer
```

### Components
- **Fine-tuned MiniLM** - Domain-specific embeddings (22M params)
- **FAISS Vector Store** - Fast similarity search
- **LFM2-1.2B-RAG** - Local LLM for answer generation
- **FastAPI Backend** - REST API at `/api/v1/mechanics/query`
- **Demo UI** - Interactive web interface at `/demo/`

---

## 📁 Project Structure

```
dnd-dm-copilot/
├── dnd_dm_copilot/
│   ├── api/                    # FastAPI backend + demo UI
│   ├── data/                   # Data processing pipelines
│   ├── evaluation/             # Evaluation pipeline (3-step)
│   ├── model/                  # LLM client (LFM2)
│   ├── training/               # Model fine-tuning
│   └── visualization/          # t-SNE analysis
├── tests/                      # Comprehensive test suite
├── data/
│   ├── processed/              # Processed datasets
│   ├── evaluation/             # Evaluation results
│   └── indices/                # FAISS indices
├── models/
│   └── sbert/                  # Fine-tuned model
├── visualizations/             # Generated plots
├── Makefile                    # Build commands
└── README.md                   # This file
```

---

## 📈 Key Findings

1. **Domain-Specific Fine-Tuning Works** - 96% improvement in Accuracy@1
2. **Improvements Are Meaningful** - 836% improvement on held-out test set
3. **Model Generalizes** - Works on D&D 5e despite training on 3.5e
4. **Complete System Built** - Production-ready RAG with demo UI
5. **Rigorous Validation** - LLM-as-a-Judge addresses midterm concerns

---

## 🔄 Reproducing the Evaluation

### Full Evaluation Pipeline
```bash
# Generate 500 questions from 5e corpus
make evaluate-questions CORPUS=data/processed/5e_corpus.json

# Run RAG pipeline
make evaluate-rag LFM2_MODEL=models/lfm2/LFM2-1.2B-RAG-Q4_0.gguf

# Judge answers with LLM
make evaluate-judge

# Results saved to:
# - data/evaluation/qa_triplets.json
# - data/evaluation/rag_results.json
# - data/evaluation/judgments.json
```

### Baseline Evaluation (No RAG)
```bash
make evaluate-baseline LFM2_MODEL=models/lfm2/LFM2-1.2B-RAG-Q4_0.gguf
make evaluate-judge-baseline
```

---

## 🤝 How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Run quality checks (`make quality-check`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Quality
```bash
make format      # Format with black
make lint        # Lint with flake8
make type-check  # Type check with mypy
make test        # Run tests
```

---

## 📚 Additional Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI assistant guidance for development
- **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** - Data source details
- **[docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)** - Technical implementation
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Project timeline
- **[EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md)** - Evaluation guide

---

## 🙏 Acknowledgments

### Tools & Frameworks
- [sentence-transformers](https://www.sbert.net/) - Embedding model training
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) - Backend API
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [DeepSeek](https://www.deepseek.com/) - LLM-as-a-Judge evaluation
- [LiquidAI LFM2](https://huggingface.co/LiquidAI/LFM2-1.2B-RAG) - Answer generation

### Data Sources
- [m0no1/dnd-mechanics-dataset](https://huggingface.co/datasets/m0no1/dnd-mechanics-dataset) - Training data
- D&D 5e SRD - Test corpus

---

## 📄 License

This project is for educational purposes as part of BU CS 506.

---

## 📧 Contact

**Garry Kuwanto** - BU CS 506 Final Project

For questions about this project, please open an issue on GitHub.
