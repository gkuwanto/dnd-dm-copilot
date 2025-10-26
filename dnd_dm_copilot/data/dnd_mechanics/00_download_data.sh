#!/bin/bash
# Download dnd-mechanics-dataset from Hugging Face

# Get old HF token to not interfere with machine environment
OLD_HF_TOKEN=$HF_TOKEN

source .env

export HF_TOKEN=$HF_TOKEN

hf download m0no1/dnd-mechanics-dataset --repo-type dataset --token $HF_TOKEN

# Set HF token back to old HF token to not interfere with machine environment
export HF_TOKEN=$OLD_HF_TOKEN