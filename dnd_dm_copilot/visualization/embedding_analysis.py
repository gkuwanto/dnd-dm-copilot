#!/usr/bin/env python3
"""
Embedding Analysis for D&D DM Copilot
Creates query-passage group visualizations using real MiniLM models and test set data.

Usage:
    uv run dnd_dm_copilot.visualization.embedding_analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available - install with: pip install sentence-transformers")

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available - install with: pip install scikit-learn")

def load_minilm_models():
    """Load baseline and fine-tuned MiniLM models"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        print("🔄 Loading baseline MiniLM model...")
        baseline_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("🔄 Loading fine-tuned model...")
        finetuned_model = SentenceTransformer('models/sbert/')
        
        print("✅ Successfully loaded both models!")
        return baseline_model, finetuned_model
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

def load_test_dataset():
    """Load D&D mechanics dataset and split it like finetune.py"""
    try:
        print("🔄 Loading D&D mechanics dataset...")
        with open('dnd-mechanics-dataset.json', 'r') as f:
            data = json.load(f)
        
        print(f"📚 Loaded {len(data)} training pairs")
        
        # Split dataset like finetune.py (80/10/10)
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Sample 30 pairs from test set
        if len(test_data) >= 30:
            test_sample = test_data[:30]
        else:
            test_sample = test_data
        
        print(f"🎯 Using {len(test_sample)} test pairs for visualization")
        return test_sample
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def generate_test_embeddings(models, test_pairs):
    """Generate embeddings for actual test set query-passage pairs"""
    if not models or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Cannot generate test embeddings - models not available")
        return None, None, None, None, None
    
    baseline_model, finetuned_model = models
    
    print("🔄 Generating embeddings for test set query-passage pairs...")
    
    # Extract queries and passages from test pairs
    queries = [pair['query'] for pair in test_pairs]
    passages = [pair['passage'] for pair in test_pairs]
    
    # Create group labels (each query-passage pair is a group)
    group_labels = [f"Group {i+1}" for i in range(len(test_pairs))]
    
    try:
        # Generate embeddings with baseline model
        print("📊 Generating baseline embeddings for test queries...")
        baseline_query_embeddings = baseline_model.encode(queries, show_progress_bar=True)
        
        print("📊 Generating baseline embeddings for test passages...")
        baseline_passage_embeddings = baseline_model.encode(passages, show_progress_bar=True)
        
        # Generate embeddings with fine-tuned model
        print("📊 Generating fine-tuned embeddings for test queries...")
        finetuned_query_embeddings = finetuned_model.encode(queries, show_progress_bar=True)
        
        print("📊 Generating fine-tuned embeddings for test passages...")
        finetuned_passage_embeddings = finetuned_model.encode(passages, show_progress_bar=True)
        
        print(f"✅ Generated embeddings for {len(queries)} test query-passage pairs")
        print(f"📏 Embedding dimension: {baseline_query_embeddings.shape[1]}")
        
        return (baseline_query_embeddings, baseline_passage_embeddings), \
               (finetuned_query_embeddings, finetuned_passage_embeddings), \
               queries, passages, group_labels
        
    except Exception as e:
        print(f"❌ Error generating test embeddings: {e}")
        return None, None, None, None, None

def create_query_passage_group_plot(baseline_embeddings, finetuned_embeddings, queries, passages, group_labels, title, filename):
    """Create t-SNE visualization showing query-passage pairs as groups"""
    if not SKLEARN_AVAILABLE:
        print(f"⚠️  Skipping {filename} - scikit-learn not available")
        return
    
    print(f"📊 Creating query-passage group plot: {title}")
    
    baseline_query_emb, baseline_passage_emb = baseline_embeddings
    finetuned_query_emb, finetuned_passage_emb = finetuned_embeddings
    
    # Combine query and passage embeddings for t-SNE
    baseline_combined = np.vstack([baseline_query_emb, baseline_passage_emb])
    finetuned_combined = np.vstack([finetuned_query_emb, finetuned_passage_emb])
    
    # Create t-SNE for both models
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(baseline_combined)//4))
    
    baseline_2d = tsne.fit_transform(baseline_combined)
    finetuned_2d = tsne.fit_transform(finetuned_combined)
    
    # Split back into query and passage coordinates
    n_pairs = len(queries)
    baseline_query_2d = baseline_2d[:n_pairs]
    baseline_passage_2d = baseline_2d[n_pairs:]
    finetuned_query_2d = finetuned_2d[:n_pairs]
    finetuned_passage_2d = finetuned_2d[n_pairs:]
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Generate colors for each group
    colors = plt.cm.tab20(np.linspace(0, 1, n_pairs))
    
    # Baseline plot
    for i in range(n_pairs):
        # Plot query and passage as connected points
        ax1.scatter(baseline_query_2d[i, 0], baseline_query_2d[i, 1], 
                   c=[colors[i]], marker='o', s=100, alpha=0.8, label=f'Query {i+1}' if i < 5 else "")
        ax1.scatter(baseline_passage_2d[i, 0], baseline_passage_2d[i, 1], 
                   c=[colors[i]], marker='s', s=100, alpha=0.8, label=f'Passage {i+1}' if i < 5 else "")
        
        # Draw line connecting query to passage
        ax1.plot([baseline_query_2d[i, 0], baseline_passage_2d[i, 0]], 
                [baseline_query_2d[i, 1], baseline_passage_2d[i, 1]], 
                color=colors[i], alpha=0.5, linewidth=1)
    
    ax1.set_title('Baseline Model\n(Query-Passage Pairs)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Fine-tuned plot
    for i in range(n_pairs):
        # Plot query and passage as connected points
        ax2.scatter(finetuned_query_2d[i, 0], finetuned_query_2d[i, 1], 
                   c=[colors[i]], marker='o', s=100, alpha=0.8, label=f'Query {i+1}' if i < 5 else "")
        ax2.scatter(finetuned_passage_2d[i, 0], finetuned_passage_2d[i, 1], 
                   c=[colors[i]], marker='s', s=100, alpha=0.8, label=f'Passage {i+1}' if i < 5 else "")
        
        # Draw line connecting query to passage
        ax2.plot([finetuned_query_2d[i, 0], finetuned_passage_2d[i, 0]], 
                [finetuned_query_2d[i, 1], finetuned_passage_2d[i, 1]], 
                color=colors[i], alpha=0.5, linewidth=1)
    
    ax2.set_title('Fine-tuned Model\n(Query-Passage Pairs)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'{title}\nQuery-Passage Pair Clustering', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ {filename} saved successfully")

def main():
    """Main function to generate query-passage group visualizations"""
    print("🎯 D&D DM Copilot - Query-Passage Group Analysis")
    print("=" * 50)
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load test dataset
    test_pairs = load_test_dataset()
    if test_pairs is None:
        print("❌ Cannot proceed without test dataset")
        return
    
    # Try to load real MiniLM models
    models = load_minilm_models()
    
    if models[0] is not None:  # Models loaded successfully
        print("\n🚀 Using REAL MiniLM models for embedding generation!")
        
        # Use actual test query-passage pairs
        result = generate_test_embeddings(models, test_pairs)
        if result[0] is not None:
            baseline_embeddings, finetuned_embeddings, queries, passages, group_labels = result
            use_real_embeddings = True
            print(f"🔢 Generated embeddings for {len(queries)} query-passage pairs")
        else:
            use_real_embeddings = False
    else:
        print("⚠️  Could not load MiniLM models")
        use_real_embeddings = False
    
    # Create visualization
    if SKLEARN_AVAILABLE and use_real_embeddings:
        print("\n📊 Creating query-passage group visualization...")
        create_query_passage_group_plot(
            baseline_embeddings, finetuned_embeddings, queries, passages, group_labels,
            'Test Set - Real MiniLM', 'query_passage_groups.png'
        )
    else:
        if not SKLEARN_AVAILABLE:
            print("⚠️  Skipping visualization - scikit-learn not available")
        if not use_real_embeddings:
            print("⚠️  Skipping visualization - models not available")
    
    print("\n" + "=" * 50)
    if use_real_embeddings:
        print("✨ Used REAL MiniLM model embeddings for authentic analysis!")
    else:
        print("⚠️  Could not generate embeddings - check model availability")
    
    print("\nGenerated files:")
    if SKLEARN_AVAILABLE and use_real_embeddings:
        print("• query_passage_groups.png - Query-passage pair clustering visualization")

if __name__ == "__main__":
    main()