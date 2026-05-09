from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

ds = load_dataset("codeparrot/codeparrot-clean-valid",
                  split="train[:20000]",
                  trust_remote_code=True)

print(f"Samples: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"\nPreview:\n{ds[0]['content'][:300]}")

with open("data/python_code.txt", "w", encoding="utf-8") as f:
    for sample in ds:
        f.write(sample["content"] + "\n\n")

size_mb = os.path.getsize("data/python_code.txt") / 1024 / 1024
print(f"Saved → data/python_code.txt ({size_mb:.1f} MB)")
