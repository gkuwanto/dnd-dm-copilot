def main():
    print("Hello from final-project!")


if __name__ == "__main__":
    main()

"""
uv run -m dnd_dm_copilot.training.finetune \
  --dataset crd3_training_pairs_high_quality.json \
  --test_dataset reddit_training_pairs.json \
  --output_dir models/crd3-dm-sbert/ \
  --batch_size 64 \
  --num_epochs 1 \
  --evaluation_steps 50 \
  --wandb_project "dnd-dm-copilot" \
  --wandb_run_name "crd3-hq-reddit-test-1epoch"
"""
