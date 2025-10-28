# D&D Dungeon Master Copilot: A Context-Aware RAG System for Improvisational Gameplay

**Course:** BU CS 506 - Final Project  
**Team:** Garry Kuwanto  
**Midterm Report:** October 27, 2024

## 📹 Midterm Presentation Video
[![Watch this video](https://img.youtube.com/vi/OPkHkA7z7tw/hqdefault.jpg)](https://youtu.be/OPkHkA7z7tw)
## 📊 Midterm Report - Preliminary Results

### Data Processing Pipeline
We successfully processed the **m0no1/dnd-mechanics-dataset** from Hugging Face, containing 40,365 D&D 3.5 mechanics question-answer pairs. Our data processing pipeline includes:

- **Train/Validation/Test Splits:** Implemented proper 80/10/10 data splits for fair evaluation
- **Data Loading Infrastructure:** Created reusable data loading utilities with proper preprocessing

### Modeling Methods
We fine-tuned the **sentence-transformers/all-MiniLM-L6-v2** model using:

- **Training Strategy:** Contrastive learning with MultipleNegativesRankingLoss
- **Framework:** sentence-transformers library in Python
- **Training Duration:** 5 epochs (2525 steps) with evaluation tracking
- **Experiment Tracking:** Weights & Biases integration for monitoring training progress

### Preliminary Results
Our fine-tuned model shows **signnificant improvements** over the baseline across all key metrics:

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| **Accuracy@1** | 34.8% | **68.2%** | **+96%** |
| **Accuracy@3** | 50.0% | **78.2%** | **+56%** |
| **Accuracy@5** | 55.5% | **82.6%** | **+49%** |
| **MRR@10** | 43.5% | **74.9%** | **+72%** |

### Visual Evidence: Query-Passage Pair Clustering
Our t-SNE visualization of query-passage pairs demonstrates the model's improved semantic understanding:

![Query-Passage Groups](visualizations/query_passage_groups.png)

**Key Observations:**
- **Baseline Model (Left):** Query-passage pairs are scattered with long connecting lines, indicating poor semantic alignment
- **Fine-tuned Model (Right):** Query-passage pairs cluster much closer together with shorter connecting lines, showing improved semantic understanding
- **Visual Improvement:** The fine-tuned model successfully brings related queries and passages into proximate regions of the embedding space
- **Quantitative Validation:** This visual evidence directly supports our quantitative metrics showing 72% improvement in MRR@10

### Training Progress
The validation MRR score shows consistent improvement throughout training, reaching convergence around 0.725 after 2,500 steps, demonstrating stable learning without overfitting.

### Key Findings
1. **Significant Performance Gains:** Our fine-tuned model achieves 96% improvement in Accuracy@1, demonstrating successful domain adaptation
2. **Visual Confirmation:** t-SNE plots provide clear visual evidence of improved query-passage semantic alignment
3. **Stable Training:** Validation metrics show consistent improvement without overfitting
4. **Reproducible Pipeline:** Complete training and evaluation pipeline ready for replication

### Preliminary Code Implementation
Our project repository contains comprehensive preliminary code demonstrating the complete pipeline:

- **`dnd_dm_copilot/training/finetune.py`** - Complete training pipeline with data loading, model fine-tuning, and evaluation
- **`dnd_dm_copilot/visualization/embedding_analysis.py`** - Visualization script for query-passage pair clustering analysis

### Next Steps
- **RAG Integration:** Build full retrieval-augmented generation system using fine-tuned embeddings
- **Dataset Expansion:** Integrate additional D&D datasets (FIREBALL, CRD3) for broader coverage
- **Qualitative Evaluation:** Create challenge set with diverse D&D scenarios for human evaluation

## Project Description

This project aims to create a D&D Dungeon Master copilot tool that acts like Cursor for code - providing intelligent, context-aware assistance to help DMs improvise on the fly. The system will use a specialized RAG pipeline to retrieve relevant information from campaign notes, game mechanics, and similar situations from other campaigns to support real-time decision making.

### Problem Statement

Dungeon Masters face constant pressure to improvise and make quick decisions during gameplay. They need to:
- Remember complex campaign lore and NPC details
- Apply game mechanics correctly in dynamic situations  
- Draw inspiration from similar scenarios in other campaigns
- Maintain narrative consistency while adapting to player choices

Current tools are either too generic (general D&D wikis) or too static (campaign notes), failing to provide context-aware assistance for real-time improvisation.

### Proposed Solution

We will create a DM copilot tool that combines:
- **Campaign-Specific Knowledge:** Current campaign notes, NPCs, locations, plot threads
- **Mechanics Database:** Rules, spells, abilities, and their contextual applications
- **Cross-Campaign Inspiration:** Similar situations from other campaigns for creative ideas
- **Context-Aware Retrieval:** Understanding of current game state and narrative context

The system will provide instant, relevant suggestions to help DMs make informed decisions on the fly.

## Project Goals

### Primary Goal (Core Scope)
Create a focused retrieval system using a single, well-curated D&D dataset to demonstrate the value of domain-specific fine-tuning. The project will focus on:
- Fine-tuning a lightweight embedding model (all-MiniLM-L6-v2) on D&D mechanics data
- Comparing retrieval performance against the base model
- Demonstrating improved semantic understanding of D&D concepts

### Core Success Criteria
- **Retrieval Quality:** Fine-tuned model shows measurable improvement in MRR/Recall@k over baseline on D&D mechanics dataset
- **Semantic Understanding:** Visual evidence of better clustering of D&D terms in embedding space
- **Reproducible Pipeline:** Complete training and evaluation pipeline that others can replicate
- **Documentation:** Clear documentation of methodology and results

### Stretch Goals (If Time Permits)
- **Multi-Source Integration:** Expand to include FIREBALL and CRD3 datasets
- **Advanced Evaluation:** Comprehensive challenge set with qualitative analysis
- **RAG System Integration:** Full retrieval-augmented generation system
- **Cross-Campaign Analysis:** Analysis of retrieval across different campaign types

## Data Collection Plan

### Core Dataset (Primary Focus)
1. **D&D Mechanics Dataset (Hugging Face):** 40,365 D&D 3.5 mechanics question-answer pairs
   - Already in (query, passage) format - minimal preprocessing needed
   - Split into 80% training, 10% validation, 10% test
   - Focus on high-quality, domain-specific content

### Stretch Goal Datasets (If Time Permits)
2. **FIREBALL Dataset (Hugging Face):** ~25,000 unique D&D sessions from Discord gameplay
3. **CRD3 Dataset (GitHub):** 398,682 turns from 159 Critical Role episodes
4. **Custom Evaluation Set:** Manually curated challenge scenarios for qualitative testing

### Simplified Data Processing
- **Core Approach:** Use existing D&D mechanics dataset with minimal preprocessing
- **Stretch Goal:** Implement LLM-enhanced preprocessing for additional datasets
- **Focus:** Quality over quantity - ensure clean, relevant training pairs

*Detailed data processing strategies are documented in [DATA_SOURCES.md](docs/DATA_SOURCES.md)*

## Modeling Plan

### Core Model (Primary Focus)
- **Base Model:** sentence-transformers/all-MiniLM-L6-v2 (22M parameters)
- **Rationale:** Lightweight, fast to train, proven performance for domain adaptation
- **Training Time:** Estimated 2-4 hours on standard hardware

### Training Approach
- **Framework:** sentence-transformers library in Python
- **Strategy:** Contrastive learning with MultipleNegativesRankingLoss
- **Process:** Fine-tune on D&D mechanics dataset to improve semantic understanding
- **Validation:** Use held-out test set for performance evaluation

### Stretch Goal Models
- **Advanced Model:** Qwen3 Embeddings 0.6B (if time permits)
- **Multi-Dataset Training:** Combine multiple D&D datasets for broader coverage
- **Advanced Techniques:** Experiment with different loss functions and training strategies

*Detailed technical implementation is documented in [TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)*

## Visualization Plan

### Core Visualizations (Required)
- **Embedding Space Analysis:** t-SNE plots showing D&D term clustering before/after fine-tuning
- **Performance Metrics:** Bar charts comparing MRR and Recall@k scores (baseline vs. fine-tuned)
- **Retrieval Examples:** 5-10 side-by-side comparisons of top-3 retrieved results

### Stretch Goal Visualizations (If Time Permits)
- **Word Clouds:** Most frequent D&D terms in the dataset
- **Data Distribution:** Visualize data across different sources
- **Advanced Clustering:** UMAP plots and interactive visualizations
- **Comprehensive Examples:** 20+ retrieval examples across different D&D concepts

## Test Plan

### Core Evaluation (Required)
- **Holdout Test Set:** 10% of dnd-mechanics-dataset (~4,000 pairs)
- **Metrics:** Mean Reciprocal Rank (MRR) and Recall@k (k=1, 3, 5)
- **Baseline Comparison:** Compare fine-tuned model against base all-MiniLM-L6-v2 model
- **Statistical Significance:** Basic t-test to determine if improvements are significant

### Stretch Goal Evaluation (If Time Permits)
- **Challenge Set:** 20-30 manually crafted diverse DM scenarios
- **Multi-Source Testing:** Evaluate retrieval across different knowledge sources
- **Advanced Statistical Analysis:** Confidence intervals, effect sizes, and comprehensive significance testing
- **Cross-Dataset Validation:** Test on FIREBALL and CRD3 datasets

## Project Timeline

### Phase 1: Data Preparation (Weeks 1-2)
- Download and explore D&D mechanics dataset
- Implement basic data preprocessing and train/val/test splits
- Set up training environment

### Phase 2: Model Training (Weeks 3-4)
- Implement fine-tuning pipeline for all-MiniLM-L6-v2
- Train model and validate performance
- Iterate on hyperparameters if needed

### Phase 3: Evaluation and Analysis (Weeks 5-6)
- Implement evaluation metrics (MRR, Recall@k)
- Run quantitative evaluation on test set
- Generate core visualizations (t-SNE plots, performance comparisons)

### Phase 4: Documentation and Stretch Goals (Weeks 7-8)
- Document results and methodology
- Create reproducible pipeline
- **If time permits:** Implement stretch goals (additional datasets, advanced evaluation)

*Detailed implementation plan is documented in [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)*

## Expected Outcomes

### Core Deliverables (Required)
1. **A fine-tuned embedding model** that shows measurable improvement in D&D-specific retrieval tasks
2. **Quantitative evaluation results** demonstrating improved MRR/Recall@k scores over baseline
3. **Visual evidence** of better semantic clustering of D&D concepts in embedding space
4. **A reproducible training pipeline** that others can use to fine-tune embedding models for other domains
5. **Clear documentation** of methodology, results, and lessons learned

### Stretch Goal Deliverables (If Time Permits)
6. **Multi-dataset integration** showing how to combine different D&D data sources
7. **Advanced evaluation** with qualitative analysis and challenge sets
8. **Comprehensive visualizations** including interactive plots and detailed case studies

This project will demonstrate the value of domain-specific fine-tuning for specialized applications, with a focus on achievable, high-quality results rather than comprehensive coverage.

---

## Additional Documentation

- **[DATA_SOURCES.md](docs/DATA_SOURCES.md)** - Detailed information about data sources and processing strategies
- **[TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)** - Technical implementation details and system architecture
- **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Detailed project timeline and implementation phases