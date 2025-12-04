# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for BU CS 506 focusing on fine-tuning domain-specific embedding models for a D&D Dungeon Master RAG system. The core objective is to improve retrieval quality by fine-tuning sentence-transformers models on D&D-specific data using contrastive learning.

**Key Achievement:** The fine-tuned model shows significant improvements over baseline:
- Accuracy@1: +96% improvement (34.8% → 68.2%)
- Accuracy@3: +56% improvement (50.0% → 78.2%)
- MRR@10: +72% improvement (43.5% → 74.9%)

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Set up environment variables (copy from env.example)
cp env.example .env
# Edit .env with your API keys (HF_TOKEN, DEEPSEEK_API_KEY, WANDB_API_KEY)
```

### Data Processing
```bash
# Process D&D mechanics dataset (m0no1/dnd-mechanics-dataset)
uv run -m dnd_dm_copilot.data.dnd_mechanics.01_create_training_pairs

# Process CRD3 dataset (Critical Role transcripts)
uv run -m dnd_dm_copilot.data.crd3.01_create_training_pairs

# Process FIREBALL dataset (Discord gameplay logs)
uv run -m dnd_dm_copilot.data.fireball.01_create_training_pairs
```

### Model Training
```bash
# Basic training with default parameters
uv run -m dnd_dm_copilot.training.finetune \
  --dataset dnd-mechanics-dataset.json \
  --output_dir models/sbert/

# Training with custom parameters
uv run -m dnd_dm_copilot.training.finetune \
  --dataset dnd-mechanics-dataset.json \
  --output_dir models/sbert/ \
  --batch_size 64 \
  --num_epochs 10 \
  --evaluation_steps 50 \
  --wandb_project "dnd-dm-copilot" \
  --wandb_run_name "experiment-1"

# Disable Weights & Biases logging
uv run -m dnd_dm_copilot.training.finetune \
  --dataset dnd-mechanics-dataset.json \
  --output_dir models/sbert/ \
  --disable_wandb
```

### Visualization
```bash
# Generate t-SNE visualization of query-passage pair clustering
uv run -m dnd_dm_copilot.visualization.embedding_analysis
```

### Code Quality
```bash
# Format code with black
uv run black dnd_dm_copilot/

# Type checking with mypy
uv run mypy dnd_dm_copilot/

# Linting with flake8
uv run flake8 dnd_dm_copilot/

# Run tests (if test files exist)
uv run pytest
```

## Architecture

### Data Processing Pipeline
The project uses multiple D&D datasets, each with specific processing requirements:

1. **D&D Mechanics Dataset** (`dnd_dm_copilot/data/dnd_mechanics/`): Pre-formatted Q&A pairs from Hugging Face (m0no1/dnd-mechanics-dataset). Simply renames fields from `instruction`/`output` to `query`/`passage`.

2. **CRD3 Dataset** (`dnd_dm_copilot/data/crd3/`): Critical Role transcripts with 398,682 dialogue turns. Processing involves extracting dialogue segments and matching with abstractive summaries to create query-passage pairs.

3. **FIREBALL Dataset** (`dnd_dm_copilot/data/fireball/`): Natural language gameplay from Discord Avrae bot logs. Processing converts player intents to queries and game outcomes to passages.

All datasets output standardized JSON format: `[{"query": "...", "passage": "..."}, ...]`

### Model Training Pipeline
Located in `dnd_dm_copilot/training/finetune.py`:

- **Base Model**: sentence-transformers/all-MiniLM-L6-v2 (22M parameters)
- **Training Strategy**: Contrastive learning with MultipleNegativesRankingLoss
- **Data Split**: 80% train, 10% validation, 10% test
- **Evaluation**: Uses InformationRetrievalEvaluator with MRR@k metrics
- **Experiment Tracking**: Integrated with Weights & Biases for logging training progress and metrics
- **Model Saving**: Automatically saves best model based on validation MRR

Key training features:
- Proper train/val/test splits with stratified sampling
- Warmup steps for learning rate scheduling
- Evaluation at configurable intervals
- Baseline comparison before fine-tuning
- Test set evaluation after training

### Visualization Pipeline
Located in `dnd_dm_copilot/visualization/embedding_analysis.py`:

Generates t-SNE visualizations comparing baseline vs fine-tuned models:
- Loads test set (uses same 80/10/10 split as training)
- Generates embeddings for query-passage pairs
- Creates side-by-side plots showing clustering improvement
- Connects query-passage pairs with lines to visualize semantic distance
- Outputs to `visualizations/query_passage_groups.png`

### Model Architecture Details
The sentence-transformers model consists of two components:
1. **Transformer Layer**: Pre-trained MiniLM-L6-v2 for encoding text
2. **Pooling Layer**: Mean pooling to create fixed-size embeddings

Fine-tuning adjusts both layers to improve D&D-specific semantic understanding.

## Important File Locations

- **Training Pipeline**: `dnd_dm_copilot/training/finetune.py` - Complete end-to-end training with evaluation
- **Data Processing**: `dnd_dm_copilot/data/{dataset_name}/01_create_training_pairs.py` - Dataset-specific preprocessing
- **Visualization**: `dnd_dm_copilot/visualization/embedding_analysis.py` - t-SNE analysis and plotting
- **Configuration**: `pyproject.toml` - Dependencies and tool configuration
- **Environment Template**: `env.example` - Required API keys and configuration

## Environment Variables

Required environment variables (set in `.env` file):

- `HF_TOKEN`: Hugging Face API token for dataset access
- `DEEPSEEK_API_KEY`: DeepSeek API key for LLM-enhanced preprocessing (optional)
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking (optional, can use `--disable_wandb`)

## Data Format

All training data uses a standardized format:
```json
[
  {
    "query": "How does Divine Smite work?",
    "passage": "Divine Smite is a Paladin class feature that allows you to expend a spell slot to deal additional radiant damage to a target hit by a melee weapon attack. The damage is 2d8 for a 1st-level spell slot, plus 1d8 for each spell level higher than 1st, to a maximum of 5d8. Against undead or fiends, you add an extra 1d8."
  }
]
```

This format is used consistently across all datasets and is expected by the training pipeline.

## Key Training Parameters

When using `finetune.py`, these parameters control the training process:

- `--batch_size`: Training batch size (default: 64). Larger batches provide more negative examples for contrastive learning.
- `--num_epochs`: Number of training epochs (default: 10)
- `--evaluation_steps`: Steps between validation evaluations (default: 50)
- `--train_ratio`: Proportion of data for training (default: 0.8)
- `--val_ratio`: Proportion of data for validation (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

## Weights & Biases Integration

The training pipeline automatically logs:
- Dataset statistics (size, splits)
- Training configuration (hyperparameters)
- Baseline model performance
- Training metrics (loss, MRR scores)
- Validation metrics during training
- Final test set results
- Performance improvements (absolute and percentage)
- Model artifacts for later retrieval

Use `--disable_wandb` flag to run without W&B logging.

## Model Output Structure

After training, the model directory contains:
- `config.json`: Model configuration
- `pytorch_model.bin`: Fine-tuned model weights
- `tokenizer_config.json`: Tokenizer configuration
- `special_tokens_map.json`: Special tokens mapping
- `modules.json`: Model architecture definition

The trained model can be loaded directly with:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("path/to/model/dir")
```

## Research Context

This is an academic research project with a focus on demonstrating the value of domain-specific fine-tuning for specialized retrieval tasks. The project prioritizes:
- Reproducible training pipelines
- Quantitative evaluation with proper train/val/test splits
- Visual evidence of improved semantic understanding
- Documentation of methodology and results

The ultimate goal is creating a DM copilot tool that provides context-aware assistance for improvisational gameplay, but the current focus is on the retrieval component.
