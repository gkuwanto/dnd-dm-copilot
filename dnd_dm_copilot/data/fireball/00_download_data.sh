#!/bin/bash
# Download fails due to rate limit, 
# But thankfully HF caches
# so we retry with a sleep
# Need to do this 20 times to get all the data

# Get old HF token to not interfere with machine environment
OLD_HF_TOKEN=$HF_TOKEN

source .env

export HF_TOKEN=$HF_TOKEN

# Download data
for i in {1..20}; do
    # Retry download
    hf download lara-martin/FIREBALL --token $HF_TOKEN --repo-type dataset
    # Sleep for 10 seconds
    sleep 120s
done

# Set HF token back to old HF token to not interfere with machine environment
export HF_TOKEN=$OLD_HF_TOKEN