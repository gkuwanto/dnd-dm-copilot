# D&D Dungeon Master Copilot: A Context-Aware RAG System for Improvisational Gameplay

**Course:** BU CS 506 - Final Project  
**Team:** Garry Kuwanto  
**Proposal Due:** September 22, 2024

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