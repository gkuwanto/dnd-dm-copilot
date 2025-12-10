# D&D Dungeon Master Copilot
## Fine-Tuning Domain-Specific Embeddings for RAG

**CS 506 Final Project - Final Presentation**  
**Garry Kuwanto**  
**December 10, 2024**

---

## What is RAG?

![What is RAG?](visualizations/what-is-rag.png)

*Image source: [Data Science Central - RAG and its evolution](https://www.datasciencecentral.com/rag-and-its-evolution/)*

**Retrieval-Augmented Generation** gives smaller, faster models access to external knowledge through context.

### Why Do We Care?
- **Efficiency**: Lightweight models + external knowledge = better performance
- **Cost-effectiveness**: No need for massive models
- **Domain-specificity**: Tailored knowledge injection
- **Real-time**: Instant retrieval vs. training data limitations

---

## The DM Problem

Being a Dungeon Master is **overwhelming**.

### Players Do Unconventional Things:
- Cast spells in creative ways
- Negotiate with unprepared NPCs
- Explore undeveloped areas
- Ask unexpected questions

### DMs Need Instant Access To:
- **Game mechanics** - How does this spell work?
- **Campaign lore** - What did that NPC say?
- **Rules clarifications** - Can they actually do that?

### Current Solutions Are Inadequate:
- **Too slow** - Searching through rulebooks
- **Too generic** - General D&D wikis

---

## Project Goals

**Build a domain-specific RAG system** with fine-tuned embeddings for D&D.

### Core Objectives:
1. **Fine-tune embeddings** on D&D-specific data
2. **Build complete RAG pipeline** (retrieval + generation)
3. **Validate rigorously** - Address concerns about meaningful improvement
4. **Demonstrate generalization** - Test on different distribution (5e vs 3.5e training)

### Success Criteria:
- Measurable improvement over baseline
- End-to-end answer quality improvement
- Reproducible, production-ready system

---

## Data Collection & Processing

### Training Data: D&D 3.5e Mechanics
- **Dataset**: m0no1/dnd-mechanics-dataset (Hugging Face)
- **Size**: 40,365 question-answer pairs
- **Format**: Query-passage pairs ready for training
- **Splits**: 80% train / 10% validation / 10% test

### Test Data: D&D 5e PDF Corpus
- **Source**: D&D 5e rulebooks (different distribution!)
- **Evaluation**: 500 questions generated from random passages
- **Purpose**: Test generalization to new content

### Key Insight
Training on 3.5e, testing on 5e → Tests if model learned **D&D concepts**, not just memorized data

---

## Model Architecture & Training

### Base Model
- **sentence-transformers/all-MiniLM-L6-v2**
- 22M parameters (lightweight and efficient)
- Pre-trained on general text

### Training Strategy
- **Contrastive learning** with MultipleNegativesRankingLoss
- Fine-tune on D&D 3.5e mechanics dataset
- 5 epochs (~2,525 training steps)
- **Weights & Biases** integration for tracking

### RAG Components
- **Retriever**: Fine-tuned MiniLM + FAISS vector store
- **Generator**: LFM2-1.2B-RAG for answer generation
- **API**: FastAPI backend with demo UI

---

## Training Results (D&D 3.5e)

### Retrieval Performance on Training Distribution

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| **Accuracy@1** | 34.8% | **68.2%** | **+96%** |
| **Accuracy@3** | 50.0% | **78.2%** | **+56%** |
| **Accuracy@5** | 55.5% | **82.6%** | **+49%** |
| **MRR@10** | 43.5% | **74.9%** | **+72%** |

### Key Achievement
**96% improvement in Accuracy@1** demonstrates successful domain adaptation

*But are these gains meaningful? We need more rigorous validation...*

---

## Addressing Midterm Feedback

### The Feedback:
> "I question whether these metric gains are truly meaningful... suggest employing additional methods to validate the effectiveness of your accuracy improvements."

### Our Response: Comprehensive Evaluation Pipeline

1. **Different Distribution** - Test on D&D 5e (trained on 3.5e)
2. **LLM-as-a-Judge** - DeepSeek evaluates actual answer quality
3. **End-to-End Evaluation** - Full RAG pipeline (retrieval + generation)
4. **Baseline Comparison** - Compare against model without RAG

### Evaluation Flow:
```
500 5e Passages → Generate Q&A → Run RAG Pipeline → LLM Judge → Metrics
```

---

## Generalization Results (D&D 5e Test Set)

### Retrieval Performance (Different Distribution)
| Metric | Fine-tuned Model |
|--------|------------------|
| **Accuracy@1** | 21.2% |
| **Accuracy@3** | 34.4% |
| **Accuracy@5** | 39.2% |
| **Accuracy@10** | 45.6% |
| **MRR@10** | 0.289 |

*Lower than training distribution - expected for cross-distribution testing*

### End-to-End Answer Quality (LLM-as-a-Judge)

| System | Correct Answers | Accuracy |
|--------|-----------------|----------|
| **Baseline (No RAG)** | 22/500 | **4.4%** |
| **RAG Pipeline** | 206/500 | **41.2%** |

## **+836% Improvement in Answer Quality!**

---

## What Does 836% Mean?

### The Numbers:
- **Baseline**: 22 correct answers out of 500 (4.4%)
- **RAG Pipeline**: 206 correct answers out of 500 (41.2%)
- **Absolute improvement**: +36.8 percentage points
- **Relative improvement**: (41.2 - 4.4) / 4.4 = **836%**

### In Plain English:
Our fine-tuned RAG system produces **9x more correct answers** than the baseline model without RAG.

### Why This Matters:
- Tests on **different distribution** (5e vs 3.5e training)
- Measures **actual answer quality**, not just retrieval
- Uses **independent judge** (DeepSeek LLM)
- **Directly addresses** midterm feedback concerns

---

## Visual Evidence: Query-Passage Clustering

![Query-Passage Groups](visualizations/query_passage_groups.png)

### Baseline Model (Left)
- Query-passage pairs scattered with **long connecting lines**
- Poor semantic alignment
- Related concepts far apart

### Fine-tuned Model (Right)
- Query-passage pairs **cluster closer together**
- **Shorter connecting lines** = better semantic understanding
- D&D-specific concepts properly grouped

---

## Live Demo: D&D DM Copilot

### Demo Interface at `localhost:8000/demo/`

**Features:**
- Interactive question input
- Adjustable passage retrieval (top_k slider)
- Example queries (Divine Smite, Flanking Rules, etc.)
- Generated answers with source attribution
- Retrieved passages with relevance scores

### Try It:
```bash
make install      # Install dependencies
make download-models # Download Dependencies
make run-api      # Start server
# Visit localhost:8000/demo/
```

*Ask "How does Divine Smite work?" and see the magic happen!*

---

## System Architecture

### Complete RAG Pipeline

```
User Query → Fine-tuned Embeddings → FAISS Search → Top-k Passages
                                                          ↓
                                         LFM2-1.2B-RAG Generator
                                                          ↓
                                              Generated Answer
```

### Components Built:
- **Fine-tuned MiniLM** - Domain-specific embeddings
- **FAISS Vector Store** - Fast similarity search
- **LFM2-1.2B-RAG** - Local LLM for generation
- **FastAPI Backend** - REST API with `/api/v1/mechanics/query`
- **Demo UI** - User-friendly interface
- **Evaluation Pipeline** - 3-step automated evaluation

---

## Reproducibility

### Training + Evaluation Setup
You need to have your own deepseek api key if you want to run the whole process, set it in .env with the following format
```
DEEPSEEK_API_KEY=
HF_TOKEN=
WANDB_API_KEY=
```

```bash
# Install dependencies
make install

# Process training data
make process-mechanics

# Train embedding model
make train

# Run evaluation
make evaluate-full CORPUS=data/processed/5e_corpus.json \
                   LFM2_MODEL=models/lfm2/model.gguf

# Start demo server
make run-api
```

### Key Files
- `dnd_dm_copilot/training/finetune.py` - Training pipeline
- `dnd_dm_copilot/evaluation/` - Complete evaluation suite
- `dnd_dm_copilot/api/` - FastAPI server with demo
- `tests/` - Comprehensive test suite

---

## Key Findings

### 1. Domain-Specific Fine-Tuning Works
**96% improvement** in Accuracy@1 on training distribution

### 2. Improvements Are Meaningful
**836% improvement** in end-to-end answer quality on held-out test set

### 3. Model Generalizes
Works on D&D 5e content despite training on 3.5e data

### 4. Complete System Built
Production-ready RAG system with demo UI and REST API

### 5. Rigorous Validation
LLM-as-a-Judge on different distribution addresses midterm concerns

---

## Summary of Results

| Evaluation | Metric | Result |
|------------|--------|--------|
| Training (3.5e) | Accuracy@1 | **+96%** improvement |
| Training (3.5e) | MRR@10 | **+72%** improvement |
| Test (5e) | Answer Quality | **+836%** improvement |
| Test (5e) | Correct Answers | **41.2% vs 4.4%** |

### Bottom Line:
Fine-tuned embeddings + RAG = **9x more correct answers** on held-out test data

---

## Future Work

### Immediate Extensions
- **Campaign Knowledge Pipeline** - Separate model for lore/notes
- **Multi-source Integration** - Combine mechanics, lore, custom content
- **Real-time DM Assistant** - Desktop/mobile app for live gameplay

### Research Directions
- **Larger embedding models** - Test Qwen3 Embeddings
- **Additional datasets** - FIREBALL, CRD3 for dialogue
- **Cross-game generalization** - Pathfinder, other TTRPGs

### Broader Impact
- Methodology applicable to **other specialized domains**
- Demonstrates value of **domain-specific fine-tuning**
- **Lightweight models** can achieve strong performance

---

## Acknowledgments

### Tools & Frameworks
- **sentence-transformers** - Embedding model training
- **FAISS** - Vector similarity search
- **FastAPI** - Backend API
- **LFM2** - Small lightweight LLM model used to generate answer
- **llama-cpp-python** - Python bindings for llama.cpp to run LFM2
- **Weights & Biases** - Experiment tracking
- **DeepSeek** - LLM-as-a-Judge evaluation

### Data Sources
- **m0no1/dnd-mechanics-dataset** - Training data
- **D&D 5e SRD** - Test corpus

---

## Questions?

### Quick Links:
- **GitHub**: https://github.com/gkuwanto/dnd-dm-copilot
- **Demo**: `localhost:8000/demo/`
- **Documentation**: See README.md

### Key Metrics to Remember:
- **96%** improvement on training data
- **836%** improvement on test data
- **41.2% vs 4.4%** correct answers

**Thank you!**

🎲 *Roll for initiative on your questions!*

