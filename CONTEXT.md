# D&D DM Copilot - Development Context

## Project Overview

**Project:** Fine-Tuning a Domain-Specific Embedding Model for a Dungeons & Dragons RAG System  
**Course:** BU CS 506 - Final Project  
**Developer:** Garry Kuwanto  

## Core Problem & Solution

**Problem:** General-purpose text embedding models lack domain-specific knowledge for D&D, leading to poor retrieval quality in RAG systems for Dungeon Masters.

**Solution:** Fine-tune a sentence-transformer model on D&D-specific data to create embeddings where semantically similar game concepts are closer in vector space.

## Project Structure

```
dnd-dm-copilot/
├── data/                    # Data processing and loading
│   ├── raw/                # Raw datasets from sources
│   ├── processed/          # Cleaned and formatted data
│   └── loaders/            # Data loading utilities
├── model/                  # Model components
│   ├── api/                # LLM API access (DeepSeek)
│   ├── embeddings/         # Embedding model components
│   └── retrieval/          # RAG retrieval system
├── train/                  # Training pipeline
│   ├── scripts/            # Training scripts
│   └── configs/            # Training configurations
├── eval/                   # Evaluation and analysis
│   ├── metrics/            # Evaluation metrics (MRR, Recall@k)
│   └── visualizations/     # t-SNE, UMAP plots
├── config/                 # Configuration files
├── tests/                  # Test suites
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
│   ├── api/                # API documentation
│   └── examples/           # Usage examples
└── main.py                 # Entry point
```

## Data Sources

1. **m0no1/dnd-mechanics-dataset** (~40k Q&A pairs)
   - Primary source for game mechanics
   - Convert to (query, passage) pairs

2. **lara-martin/FIREBALL** (Avrae bot logs)
   - Natural language player interactions
   - Query: player intent, Passage: game outcome

3. **microsoft/crd3** (Critical Role D&D dataset from Hugging Face)
   - 398,682 dialogue turns from 159 Critical Role episodes
   - D&D narrative and lore context with abstractive summaries
   - Query: dialogue turns, Passage: summary chunks

4. **Custom Lorebook** (Generated from campaign notes)
   - Evaluation corpus for retrieval testing
   - Not used for training

## Technical Stack

- **Base Model:** sentence-transformers/all-MiniLM-L6-v2
- **Framework:** sentence-transformers library
- **Training:** Contrastive learning with MultipleNegativesRankingLoss
- **LLM API:** DeepSeek V3.1 for data preprocessing
- **Evaluation:** MRR, Recall@k metrics

## Development Workflow

### Phase 1: Data Processing (Weeks 1-3)
- Set up data access and download
- Implement LLM preprocessing pipeline
- Create (query, passage) pair generation
- Quality validation and train/val/test splits

### Phase 2: Model Training (Weeks 4-6)
- Set up sentence-transformers training
- Implement contrastive learning pipeline
- Train and validate embedding model
- Compare against baseline models

### Phase 3: Evaluation (Weeks 7-8)
- Implement evaluation metrics
- Create challenge set for qualitative testing
- Generate visualizations (t-SNE, UMAP)
- Statistical significance testing

### Phase 4: Integration (Weeks 9-10)
- Integrate with RAG system
- End-to-end testing
- Documentation and presentation

## Key Development Files

### Data Module (`data/`)
- `loaders/dataset_loader.py` - Unified data loading interface
- `loaders/dnd_mechanics.py` - D&D mechanics dataset loader
- `loaders/fireball.py` - FIREBALL dataset loader
- `loaders/crd3.py` - CRD3 dataset loader
- `processed/query_passage_pairs.py` - Training data format

### Model Module (`model/`)
- `api/deepseek_client.py` - DeepSeek API integration
- `retrieval/rag_system.py` - Multi-source RAG system

### Training Module (`train/`)
- `scripts/train_embeddings.py` - Main training script
- `configs/training_config.yaml` - Training hyperparameters

### Evaluation Module (`eval/`)
- `metrics/retrieval_metrics.py` - MRR, Recall@k implementation
- `metrics/statistical_tests.py` - Significance testing
- `visualizations/embedding_plots.py` - t-SNE, UMAP visualizations
- `visualizations/word_cloud.py` - Word Cloud visualization

## Success Criteria

1. **Quantitative:** Fine-tuned model achieves statistically significant improvement in MRR/Recall@k over baseline
2. **Qualitative:** Better retrieval quality on D&D-specific challenge set
3. **Visual:** Coherent semantic clusters in embedding space for D&D terms
4. **Reproducible:** Complete training pipeline with documented results

## Environment Setup

```bash
# Install dependencies
uv sync

# Set up environment variables
export DEEPSEEK_API_KEY="your_api_key_here"

# Run training
python train/scripts/train_embeddings.py --config config/training_config.yaml

# Run evaluation
python eval/metrics/retrieval_metrics.py --model_path models/fine_tuned_model
```

## Next Steps

1. Set up data access and download scripts
2. Implement LLM preprocessing pipeline
3. Create training data format and loaders
4. Set up sentence-transformers training environment
5. Implement evaluation metrics and visualization tools