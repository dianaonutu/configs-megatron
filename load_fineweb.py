import os
from datasets import load_dataset

project_space = os.environ.get("PROJECT_SPACE", os.getcwd())
cache_dir = os.path.join(project_space, ".hf_cache_dir")  
output_path = os.path.join(project_space, "datasets", "FineWeb", "raw", "fineweb-10BT.jsonl")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if not os.path.exists(output_path):
    shard = "sample-10BT"
    dataset = load_dataset("HuggingFaceFW/fineweb", shard, cache_dir=cache_dir, split="train")
    dataset.to_json(output_path)
else:
    print(f"Dataset already exists at: {output_path}")
