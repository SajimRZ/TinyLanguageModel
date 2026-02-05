from datasets import load_dataset
import json
import pathlib

# Correct dataset load
ds = load_dataset("neifuisan/Neuro-sama-QnA")["train"]

cur_dir = pathlib.Path.cwd()
parent_dir =  cur_dir.parent
output_path = cur_dir / "data" / "raw" / "neuroSama.jsonl"



with open(output_path, "w", encoding="utf-8") as f_out:
    for item in ds:
        q = item.get("instruction", "").strip()
        a = item.get("output", "").strip()

        if "Neuro" in a:
            a = a.replace("Neuro", "Lumi")
        if "Neuro" in q:
            q = q.replace("Neuro", "Lumi")
        
        if not q or not a:
            continue   # skip empty

        entry = {"dialouge": [{"user": q}, {"lumi": a}]}
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(ds[0])
print(f"Done. Saved {len(ds)} rows to {output_path}")
