# Technical Implementation Details

This document contains detailed technical information about the D&D DM Copilot system implementation.

## System Architecture

### DM Copilot Interface
The system provides real-time assistance to DMs through a simple query interface:

**Input:** DM's question or situation description
- Example: "Player wants to cast fireball in a crowded tavern"
- Example: "What should happen when the party finds the missing caravan?"
- Example: "How do I handle a player trying to seduce the dragon?"

### Retrieval-Focused Architecture
1. **Query Processing:** Parse DM query and extract key concepts
2. **Multi-Source Search:** Use fine-tuned embedding model to search across all knowledge bases:
   - Campaign-specific knowledge (NPCs, locations, plot threads)
   - Mechanics database (rules, spells, abilities)
   - Cross-campaign inspiration (similar situations, outcomes)
   - Narrative techniques (storytelling methods, character development)
3. **Context-Aware Ranking:** Fine-tuned model prioritizes results based on D&D-specific context
4. **Retrieval Output:** Return top-k most relevant information chunks for each source

### Fixed Generation Component
- **Generation Model:** Use existing LLM (e.g., GPT-4, Claude) for text generation
- **Input:** Retrieved information chunks from fine-tuned retrieval system
- **Output:** Coherent suggestions combining information from multiple sources
- **Focus:** This project focuses on improving retrieval quality, not generation

## LLM-Enhanced Data Preprocessing

### DeepSeek Integration Strategy
We will use DeepSeek V3.1 (or similar LLM) to enhance our training data quality through several preprocessing techniques:

#### 1. **Semantic Standardization**
- **Goal:** Convert informal dialogue into standardized D&D terminology
- **Process:** Use DeepSeek to rewrite player actions using proper game mechanics language
- **Example:** "I hit the orc with my sword" → "The fighter makes a melee weapon attack with their longsword against the orc"

#### 2. **Context Enrichment**
- **Goal:** Add missing game context to make passages more informative
- **Process:** DeepSeek generates additional context about rules, mechanics, and consequences
- **Example:** Raw dialogue + DeepSeek enhancement = "The paladin uses Divine Smite, adding 2d8 radiant damage to their attack roll of 18, which hits the undead skeleton's AC of 13"

#### 3. **Query Generation**
- **Goal:** Create diverse, natural questions from dialogue content
- **Process:** DeepSeek generates multiple question variations for each dialogue segment
- **Examples:**
  - "What happens when a paladin uses Divine Smite?"
  - "How does Divine Smite work against undead?"
  - "What's the damage calculation for Divine Smite?"

#### 4. **Quality Filtering**
- **Goal:** Remove low-quality or irrelevant training pairs
- **Process:** DeepSeek scores dialogue-passage pairs for relevance and D&D-specific content
- **Criteria:** Keep only pairs with high D&D relevance scores

#### 5. **Data Augmentation**
- **Goal:** Increase training data diversity
- **Process:** DeepSeek generates paraphrases and alternative phrasings
- **Example:** "Cast fireball" → ["Use fireball spell", "Launch fireball", "Invoke fireball magic"]

### Implementation Plan
1. **Phase 1:** Set up DeepSeek API integration
2. **Phase 2:** Develop preprocessing prompts and validation
3. **Phase 3:** Process all datasets with LLM enhancement:
   - **Training Data:** CRD3 dataset, FIREBALL dataset, D&D mechanics dataset
   - **Evaluation Data:** Asylum Tapes campaign, Personal campaign notes (if available - stretch goal)
4. **Phase 4:** Quality assessment and iterative improvement

## Modeling Details

### Base Model Selection
- **Options:** 
  - sentence-transformers/all-MiniLM-L6-v2 (efficient, proven performance)
  - Qwen3 Embeddings 0.6B (larger, potentially more capable)
- **Selection Criteria:** Will evaluate both models during initial experimentation phase
- **Rationale:** MiniLM offers proven efficiency and speed, while Qwen3 provides larger capacity for complex D&D semantics

### Framework
- **Library:** sentence-transformers in Python
- **Training Strategy:** Contrastive learning approach
- **Loss Function:** MultipleNegativesRankingLoss

### Training Process
1. **Data Preparation:** Convert all sources into (query, passage) pairs for contrastive learning
2. **Contrastive Learning:** Train model to maximize cosine similarity between correct query-passage pairs while minimizing similarity to incorrect pairs in the same batch
3. **Fine-tuning:** Use standard sentence-transformer fine-tuning pipeline
4. **Validation:** Monitor training loss and validation metrics
5. **Retrieval Evaluation:** Test retrieval quality on D&D-specific test sets

### Generation Component (Fixed)
- **Model:** Use existing LLM (GPT-4, Claude, or similar) for text generation
- **Input:** Retrieved passages from fine-tuned embedding model
- **Role:** Convert retrieved information into coherent DM suggestions
- **Focus:** This project does not modify or train the generation component

## Data Processing Details

### Asylum Tapes Campaign Preprocessing
**Specific approach for Reddit campaign data:**
1. **Session Parsing:** Extract structured information from each Reddit post
2. **Character Extraction:** Use DeepSeek to identify and profile all NPCs mentioned
3. **Location Mapping:** Document all locations, their descriptions, and current state
4. **Plot Thread Analysis:** Track storylines, player decisions, and unresolved mysteries
5. **Query Generation:** Create realistic DM queries that would reference this campaign knowledge
6. **Knowledge Chunking:** Break down campaign information into searchable, relevant chunks

**Example Processing:**
- **Input:** Raw Reddit post about Session 01
- **DeepSeek Processing:** Extract NPCs, locations, events, plot points
- **Output:** Structured knowledge chunks like "Dr. Marcus Webb: Former asylum director, driven mad by experiments, knows about the hidden basement..."

## Technical Considerations

### Computational Requirements
- **Training:** GPU recommended for fine-tuning (Google Colab Pro or local GPU)
- **Storage:** [TBD] GB for datasets and model checkpoints
- **Memory:** Sufficient RAM for batch processing of text data

### Potential Challenges
- **Data Quality:** Ensuring consistent formatting across different data sources
- **Domain Specificity:** Balancing general language understanding with D&D-specific knowledge
- **Evaluation:** Creating meaningful test sets that capture real-world usage

### Risk Mitigation
- **Fallback Plan:** If fine-tuning doesn't show significant improvement, focus on comprehensive analysis of why and document lessons learned
- **Scope Management:** Prioritize core functionality over advanced features
- **Data Backup:** Maintain multiple copies of processed datasets

## Performance Metrics

### Quantitative Metrics
- **Mean Reciprocal Rank (MRR):** Measures the rank of the first relevant result
- **Recall@k (k=1, 3, 5):** Measures how many relevant results are in the top-k
- **Precision@k:** Measures the proportion of relevant results in the top-k

### Qualitative Metrics
- **Relevance Scoring:** Human evaluation of retrieved results
- **Context Appropriateness:** How well results match the given game state
- **Source Diversity:** How well the model retrieves from different knowledge sources

### Statistical Testing
- **Significance Testing:** Use appropriate statistical tests to determine if improvements are significant
- **Confidence Intervals:** Report confidence intervals for all metrics
- **Cross-Validation:** Use k-fold cross-validation for robust evaluation
