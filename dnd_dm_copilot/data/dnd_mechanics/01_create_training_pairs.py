# Since this dataset already in the correct format, we can just use it

import json
import os
from typing import Dict, List

import dotenv
import huggingface_hub
from datasets import Dataset

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def load_dnd_mechanics_dataset() -> List[Dataset]:
    """Load dnd-mechanics-dataset from Hugging Face"""
    try:
        from datasets import load_dataset

        dataset = load_dataset("m0no1/dnd-mechanics-dataset", split="train")
        return dataset

    except Exception as e:
        print(f"Error loading dnd-mechanics-dataset: {e}")
        return []


def preprocess_dataset(dataset: List[Dict]) -> List[Dict]:
    # rename instruction to query
    # rename output to passage

    processed_dataset = []

    for item in dataset:
        processed_dataset.append(
            {"query": item["instruction"], "passage": item["output"]}
        )

    return processed_dataset


def main():
    """Main function"""
    dataset = load_dnd_mechanics_dataset()
    print(f"Loaded {len(dataset)} pairs")
    processed_dataset = preprocess_dataset(dataset)
    print(f"Processed {len(processed_dataset)} pairs")

    # Save to file
    with open("dnd-mechanics-dataset.json", "w") as f:
        json.dump(processed_dataset, f)

    print(f"Saved to dnd-mechanics-dataset.json")

    huggingface_hub.create_repo(
        "garrykuwanto/dnd-mechanics-dataset",
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
    )

    # Upload to garrykuwanto/dnd-mechanics-dataset
    huggingface_hub.upload_file(
        path_or_fileobj="dnd-mechanics-dataset.json",
        path_in_repo="dnd-mechanics-dataset.json",
        repo_id="garrykuwanto/dnd-mechanics-dataset",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print(f"Uploaded to garrykuwanto/dnd-mechanics-dataset")


if __name__ == "__main__":
    main()
