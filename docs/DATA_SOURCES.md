# Data Sources and Processing Details

This document contains detailed information about the data sources used for training and evaluation of the D&D DM Copilot system.

## Training Data Sources

### Source 1: m0no1/dnd-mechanics-dataset (Hugging Face)
- **Content:** 40,365 D&D 3.5 mechanics question-answer pairs
- **Purpose:** **Mechanics Database** - Core rules and mechanics for DM reference
- **Processing:** Convert into DM-focused (situation, query, answer) triplets using DeepSeek V3.1:
  - **DM Context:** Generate scenarios where DMs need to apply these rules
  - **Quick Reference:** Format answers for fast DM decision-making
- **Example:** 
  - Situation: "Player wants to cast fireball in a crowded tavern"
  - Query: "What are the rules for fireball in enclosed spaces?"
  - Answer: "Fireball creates a 20-foot radius sphere. In enclosed spaces, it fills the available space and may cause structural damage..."

### Source 2: lara-martin/FIREBALL (Hugging Face)
- **Content:** ~25,000 unique D&D sessions from Discord gameplay with Avrae bot
- **Purpose:** **Cross-Campaign Inspiration** - Real gameplay scenarios for DM reference
- **Processing:** Extract (situation, outcome, lessons) triplets from Avrae logs:
  - **Situation Analysis:** Identify interesting gameplay moments and DM decisions
  - **Outcome Documentation:** Record what happened and why
  - **DM Lessons:** Extract insights for future similar situations
- **Example:** 
  - Situation: "Party encounters a dragon in a cave, wizard has fireball prepared"
  - Outcome: "DM ruled fireball ineffective due to dragon's fire immunity, wizard had to improvise"
  - Lesson: "Always consider creature immunities when players use elemental spells"

### Source 3: RevanthRameshkumar/CRD3 (GitHub)
- **Content:** 398,682 turns from 159 Critical Role episodes with abstractive summaries
- **Purpose:** **Narrative Inspiration** - High-quality storytelling examples and DM techniques
- **Processing:** LLM-enhanced preprocessing with DeepSeek V3.1:
  - **DM Technique Extraction:** Identify effective storytelling and improvisation techniques
  - **Narrative Pattern Recognition:** Find common story beats and how they're handled
  - **Character Interaction Analysis:** Study how NPCs are portrayed and developed
- **Example:** 
  - Situation: "Party meets a mysterious NPC in a tavern"
  - Technique: "Mercer used specific voice, mannerisms, and backstory hints to create intrigue"
  - Application: "For mysterious NPCs, establish 2-3 distinctive traits and one secret to reveal"

## Evaluation Data Sources

### Source 4: The Asylum Tapes Campaign (Reddit)
- **Content:** 9 detailed session logs from r/TalesFromDrexlor "Asylum Tapes" campaign
- **Purpose:** **Evaluation Corpus** - Real campaign data for testing retrieval quality on campaign-specific queries
- **Processing:** LLM-enhanced preprocessing with DeepSeek V3.1:
  - **Session Analysis:** Extract key events, NPCs, locations, and plot developments from each session
  - **Character Database:** Build profiles of NPCs, their motivations, relationships, and secrets
  - **Location Index:** Document places, descriptions, history, and current state
  - **Plot Thread Extraction:** Identify ongoing storylines, player goals, and unresolved mysteries
  - **Query Generation:** Create realistic DM queries that would reference this campaign knowledge
- **Example:**
  - Query: "What do we know about the mysterious stranger in the asylum?"
  - Result: "Dr. Marcus Webb: Former asylum director, driven mad by experiments, knows about the hidden basement, has a connection to the missing patients..."
- **Data Volume:** [TBD] - Estimated 50-100 campaign-specific knowledge chunks after processing
- **Usage:** **Evaluation only** - not used for training the embedding model

### Source 5: Personal Campaign Notes (Stretch Goal)
- **Content:** [TBD] - Notes from campaigns and one-shots that the author is part of
- **Purpose:** **Evaluation Corpus** - Personal campaign data for testing retrieval quality on authentic DM scenarios
- **Processing:** LLM-enhanced preprocessing with DeepSeek V3.1:
  - **Note Structure Analysis:** Extract NPCs, locations, plot points, and player decisions from personal notes
  - **Campaign Timeline Mapping:** Track story progression and character development
  - **DM Decision Documentation:** Record how decisions were made and their outcomes
  - **Player Interaction Analysis:** Study how players responded to different scenarios
  - **Query Generation:** Create queries based on real situations encountered during gameplay
- **Example:**
  - Query: "What happened when the party encountered the mysterious merchant in the forest?"
  - Result: "Elderwood Merchant: Sells magical items, knows about the ancient druid grove, has a connection to the fey realm, players were suspicious of his true motives..."
- **Data Volume:** [TBD] - Estimated 20-50 personal campaign knowledge chunks
- **Usage:** **Evaluation only** - not used for training the embedding model
- **Status:** Stretch goal - will be included if time permits and data is available

## Data Volume Summary

**Training Data Sources:**
- **Source 1:** D&D mechanics dataset (~40,365 pairs)
- **Source 2:** FIREBALL dataset (~25,000 pairs) 
- **Source 3:** CRD3 dataset (~398,682 turns)
- **Total Training Pairs:** [TBD] (estimated 50,000+ after processing)

**Evaluation Data Sources:**
- **Source 4:** Asylum Tapes campaign (~50-100 knowledge chunks)
- **Source 5:** Personal campaign notes (~20-50 knowledge chunks, stretch goal)
- **Holdout Test Set:** 15% of dnd-mechanics-dataset (~6,000 pairs)
- **Challenge Set:** 20-30 manually crafted diverse questions

## Data Processing Pipeline

### LLM-Enhanced Preprocessing
All data sources will be processed using DeepSeek V3.1 to improve quality and consistency:

1. **Semantic Standardization:** Convert informal language to standardized D&D terminology
2. **Context Enrichment:** Add missing game context to make passages more informative
3. **Query Generation:** Create diverse, natural questions from content
4. **Quality Filtering:** Remove low-quality or irrelevant training pairs
5. **Data Augmentation:** Generate paraphrases and alternative phrasings

### Data Format
All processed data will be converted into (query, passage) pairs for contrastive learning:
- **Query:** DM's question or situation description
- **Passage:** Relevant information chunk from knowledge base
- **Context:** Additional metadata about source, type, and relevance

### Quality Assurance
- **Validation:** Manual review of sample processed data
- **Consistency Checks:** Ensure formatting consistency across sources
- **Relevance Scoring:** Use LLM to score query-passage pairs for relevance
- **Iterative Improvement:** Refine processing based on quality assessment

## Data Access and Storage

### Data Sources
- **Hugging Face:** m0no1/dnd-mechanics-dataset, lara-martin/FIREBALL
- **GitHub:** RevanthRameshkumar/CRD3
- **Reddit:** r/TalesFromDrexlor Asylum Tapes campaign posts
- **Personal:** Campaign notes and one-shot experiences

### Storage Requirements
- **Raw Data:** [TBD] GB for all source datasets
- **Processed Data:** [TBD] GB for cleaned and structured data
- **Model Checkpoints:** [TBD] GB for training checkpoints
- **Total Estimated:** [TBD] GB

### Data Privacy and Ethics
- **Public Data:** All training sources are publicly available
- **Personal Data:** Personal campaign notes will be anonymized
- **Attribution:** Proper attribution given to all data sources
- **Usage Rights:** Ensure compliance with data source terms of use
