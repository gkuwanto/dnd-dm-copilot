.PHONY: help install test test-api test-eval test-data test-model test-all lint format type-check quality-check clean dev-setup run-api evaluate-full evaluate-questions evaluate-rag evaluate-judge

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)D&D DM Copilot - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(NC) %s\n", $$1, $$2}'

# ==================== Setup & Installation ====================

install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	uv sync
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

dev-setup: install ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from env.example...$(NC)"; \
		cp env.example .env; \
		echo "$(YELLOW)⚠ Please edit .env with your API keys$(NC)"; \
	fi
	@echo "$(GREEN)✓ Development environment ready$(NC)"

# ==================== Testing ====================

test: ## Run all tests with coverage
	@echo "$(BLUE)Running all tests...$(NC)"
	uv run pytest --cov=dnd_dm_copilot --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

test-api: ## Run API tests only
	@echo "$(BLUE)Running API tests...$(NC)"
	uv run pytest tests/api/ -v
	@echo "$(GREEN)✓ API tests complete$(NC)"

test-eval: ## Run evaluation tests only
	@echo "$(BLUE)Running evaluation tests...$(NC)"
	uv run pytest tests/evaluation/ -v
	@echo "$(GREEN)✓ Evaluation tests complete$(NC)"

test-data: ## Run data processing tests only
	@echo "$(BLUE)Running data tests...$(NC)"
	uv run pytest tests/data/ -v
	@echo "$(GREEN)✓ Data tests complete$(NC)"

test-model: ## Run model tests only
	@echo "$(BLUE)Running model tests...$(NC)"
	uv run pytest tests/model/ -v
	@echo "$(GREEN)✓ Model tests complete$(NC)"

test-all: ## Run all tests in parallel (fast)
	@echo "$(BLUE)Running all tests in parallel...$(NC)"
	uv run pytest -n auto --cov=dnd_dm_copilot
	@echo "$(GREEN)✓ All tests complete$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	uv run pytest-watch

# ==================== Code Quality ====================

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	uv run black dnd_dm_copilot/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run flake8 linting
	@echo "$(BLUE)Running linting...$(NC)"
	uv run flake8 dnd_dm_copilot/ tests/
	@echo "$(GREEN)✓ Linting passed$(NC)"

type-check: ## Run mypy type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	uv run mypy dnd_dm_copilot/
	@echo "$(GREEN)✓ Type checking passed$(NC)"

quality-check: format lint type-check test ## Run all quality checks (format, lint, type-check, test)
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

# ==================== Data Processing ====================

process-mechanics: ## Process D&D mechanics dataset
	@echo "$(BLUE)Processing mechanics dataset...$(NC)"
	uv run -m dnd_dm_copilot.data.dnd_mechanics.01_create_training_pairs
	@echo "$(GREEN)✓ Mechanics dataset processed$(NC)"

process-crd3: ## Process CRD3 dataset (Critical Role transcripts)
	@echo "$(BLUE)Processing CRD3 dataset...$(NC)"
	uv run -m dnd_dm_copilot.data.crd3.01_create_training_pairs
	@echo "$(GREEN)✓ CRD3 dataset processed$(NC)"

process-fireball: ## Process FIREBALL dataset (Discord gameplay logs)
	@echo "$(BLUE)Processing FIREBALL dataset...$(NC)"
	uv run -m dnd_dm_copilot.data.fireball.01_create_training_pairs
	@echo "$(GREEN)✓ FIREBALL dataset processed$(NC)"

process-all: process-mechanics process-crd3 process-fireball ## Process all datasets
	@echo "$(GREEN)✓ All datasets processed$(NC)"

# ==================== Model Training ====================

train: ## Train embedding model (default settings)
	@echo "$(BLUE)Training embedding model...$(NC)"
	uv run -m dnd_dm_copilot.training.finetune \
		--dataset data/processed/dnd-mechanics-dataset.json \
		--output_dir models/sbert/
	@echo "$(GREEN)✓ Training complete$(NC)"

train-custom: ## Train with custom parameters (edit this target as needed)
	@echo "$(BLUE)Training with custom parameters...$(NC)"
	uv run -m dnd_dm_copilot.training.finetune \
		--dataset data/processed/dnd-mechanics-dataset.json \
		--output_dir models/sbert/ \
		--batch_size 64 \
		--num_epochs 10 \
		--evaluation_steps 50 \
		--wandb_project "dnd-dm-copilot" \
		--wandb_run_name "custom-run"
	@echo "$(GREEN)✓ Training complete$(NC)"

visualize: ## Generate t-SNE visualization of embeddings
	@echo "$(BLUE)Generating embedding visualization...$(NC)"
	uv run -m dnd_dm_copilot.visualization.embedding_analysis
	@echo "$(GREEN)✓ Visualization saved to visualizations/$(NC)"

# ==================== Evaluation Pipeline ====================

build-index: ## Build FAISS index from corpus
	@echo "$(BLUE)Building FAISS index...$(NC)"
	@if [ -z "$(CORPUS)" ]; then \
		echo "$(RED)Error: CORPUS not specified$(NC)"; \
		echo "Usage: make build-index CORPUS=data/processed/corpus.json"; \
		exit 1; \
	fi
	uv run python -m scripts.build_index \
		--corpus $(CORPUS) \
		--model models/mechanics-retrieval \
		--output data/indices/mechanics/
	@echo "$(GREEN)✓ FAISS index built: data/indices/mechanics/$(NC)"

evaluate-questions: ## Step 1: Generate evaluation questions from corpus
	@echo "$(BLUE)Generating evaluation questions (50 concurrent requests)...$(NC)"
	@if [ -z "$(CORPUS)" ]; then \
		echo "$(RED)Error: CORPUS not specified$(NC)"; \
		echo "Usage: make evaluate-questions CORPUS=data/processed/5e_corpus.json"; \
		exit 1; \
	fi
	uv run -m dnd_dm_copilot.evaluation.generate_questions \
		--corpus $(CORPUS) \
		--output data/evaluation/qa_triplets.json \
		--n_samples 500 \
		--max_concurrent 50 \
		--skip_errors
	@echo "$(GREEN)✓ Questions generated: data/evaluation/qa_triplets.json$(NC)"

evaluate-rag: ## Step 2: Run RAG pipeline on questions
	@echo "$(BLUE)Running RAG pipeline...$(NC)"
	@if [ -z "$(LFM2_MODEL)" ]; then \
		echo "$(RED)Error: LFM2_MODEL not specified$(NC)"; \
		echo "Usage: make evaluate-rag LFM2_MODEL=path/to/model.gguf"; \
		exit 1; \
	fi
	uv run -m dnd_dm_copilot.evaluation.run_rag_pipeline \
		--qa_triplets data/evaluation/qa_triplets.json \
		--model_path models/sbert/ \
		--index_path data/indices/mechanics/ \
		--llm_model_path $(LFM2_MODEL) \
		--output data/evaluation/rag_results.json \
		--skip_errors
	@echo "$(GREEN)✓ RAG results saved: data/evaluation/rag_results.json$(NC)"

evaluate-judge: ## Step 3: Judge answers with LLM
	@echo "$(BLUE)Judging generated answers...$(NC)"
	uv run -m dnd_dm_copilot.evaluation.judge_answers \
		--results data/evaluation/rag_results.json \
		--output data/evaluation/judgments.json \
		--skip_errors
	@echo "$(GREEN)✓ Judgments saved: data/evaluation/judgments.json$(NC)"

evaluate-full: ## Run complete evaluation pipeline (all 3 steps)
	@echo "$(BLUE)Running full evaluation pipeline...$(NC)"
	@if [ -z "$(CORPUS)" ] || [ -z "$(LFM2_MODEL)" ]; then \
		echo "$(RED)Error: Required parameters not specified$(NC)"; \
		echo "Usage: make evaluate-full CORPUS=data/processed/5e_corpus.json LFM2_MODEL=path/to/model.gguf"; \
		exit 1; \
	fi
	@$(MAKE) evaluate-questions CORPUS=$(CORPUS)
	@$(MAKE) evaluate-rag LFM2_MODEL=$(LFM2_MODEL)
	@$(MAKE) evaluate-judge
	@echo "$(GREEN)✓ Full evaluation complete!$(NC)"
	@echo "$(YELLOW)Results:$(NC)"
	@echo "  - Questions: data/evaluation/qa_triplets.json"
	@echo "  - RAG Results: data/evaluation/rag_results.json"
	@echo "  - Judgments: data/evaluation/judgments.json"

# ==================== API Server ====================

run-api: ## Start FastAPI development server
	@echo "$(BLUE)Starting API server...$(NC)"
	uv run uvicorn dnd_dm_copilot.api.main:app --reload --host 0.0.0.0 --port 8000
	@echo "$(GREEN)✓ API running at http://localhost:8000$(NC)"

run-api-prod: ## Start FastAPI production server
	@echo "$(BLUE)Starting production API server...$(NC)"
	uv run uvicorn dnd_dm_copilot.api.main:app --host 0.0.0.0 --port 8000 --workers 4
	@echo "$(GREEN)✓ Production API running$(NC)"

# ==================== Utility Commands ====================

clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned up$(NC)"

clean-data: ## Clean processed data (WARNING: deletes processed datasets)
	@echo "$(YELLOW)⚠ This will delete all processed data$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/*; \
		rm -rf data/evaluation/*; \
		rm -rf data/indices/*; \
		echo "$(GREEN)✓ Data cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-models: ## Clean trained models (WARNING: deletes trained models)
	@echo "$(YELLOW)⚠ This will delete all trained models$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/sbert/*; \
		rm -rf models/crd3-dm-sbert/*; \
		echo "$(GREEN)✓ Models cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

status: ## Show project status (git, env, data, models)
	@echo "$(BLUE)Project Status$(NC)"
	@echo ""
	@echo "$(YELLOW)Git Status:$(NC)"
	@git status --short || echo "Not a git repository"
	@echo ""
	@echo "$(YELLOW)Environment:$(NC)"
	@if [ -f .env ]; then echo "✓ .env file exists"; else echo "✗ .env file missing"; fi
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@if [ -d data/processed ] && [ "$$(ls -A data/processed 2>/dev/null)" ]; then \
		echo "✓ Processed data exists: $$(ls data/processed | wc -l) files"; \
	else \
		echo "✗ No processed data"; \
	fi
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@if [ -d models/sbert ] && [ "$$(ls -A models/sbert 2>/dev/null)" ]; then \
		echo "✓ Fine-tuned model exists"; \
	else \
		echo "✗ No fine-tuned model"; \
	fi
	@echo ""
	@echo "$(YELLOW)Tests:$(NC)"
	@echo "  API tests: $$(find tests/api -name 'test_*.py' | wc -l) files"
	@echo "  Evaluation tests: $$(find tests/evaluation -name 'test_*.py' | wc -l) files"

# ==================== Quick Start Workflows ====================

quickstart: dev-setup test ## Quick start: setup environment and run tests
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run 'make process-mechanics' to process data"
	@echo "  3. Run 'make train' to train model"
	@echo "  4. Run 'make evaluate-full CORPUS=... LFM2_MODEL=...' to evaluate"

ci-check: format lint type-check test ## Run all CI checks locally
	@echo "$(GREEN)✓ All CI checks passed!$(NC)"

# ==================== Model Publishing ====================

push-model: ## Push model to Hugging Face Hub (requires HF_TOKEN)
	@echo "$(BLUE)Pushing model to Hugging Face Hub...$(NC)"
	@if [ -z "$(REPO_ID)" ]; then \
		echo "$(RED)Error: REPO_ID not specified$(NC)"; \
		echo "Usage: make push-model REPO_ID=username/model-name"; \
		exit 1; \
	fi
	uv run python -m scripts.push_to_hub \
		--model_path models/mechanics-retrieval \
		--repo_id $(REPO_ID)
	@echo "$(GREEN)✓ Model pushed to https://huggingface.co/$(REPO_ID)$(NC)"

push-model-private: ## Push model as private repository
	@echo "$(BLUE)Pushing private model to Hugging Face Hub...$(NC)"
	@if [ -z "$(REPO_ID)" ]; then \
		echo "$(RED)Error: REPO_ID not specified$(NC)"; \
		echo "Usage: make push-model-private REPO_ID=username/model-name"; \
		exit 1; \
	fi
	uv run python -m scripts.push_to_hub \
		--model_path models/mechanics-retrieval \
		--repo_id $(REPO_ID) \
		--private
	@echo "$(GREEN)✓ Private model pushed to https://huggingface.co/$(REPO_ID)$(NC)"

# ==================== Documentation ====================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation files:$(NC)"
	@echo "  README.md - Main documentation"
	@echo "  CLAUDE.md - AI assistant guidance"
	@echo "  docs/IMPLEMENTATION_PLAN.md - Implementation plan"
	@echo "  docs/TECHNICAL_DETAILS.md - Technical details"
	@echo "  docs/DATA_SOURCES.md - Data source information"
