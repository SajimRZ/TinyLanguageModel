# Data/Tiny.py
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer
import re

class BookDataset:


    def __init__(self, skip=0, take=10000, val_split=0.1, train_mask = None):
        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split

        print(f"Loading Books: skip={skip}, take={take}")

        texts = self._load_texts()

        print(f"Total samples of book lines: {len(texts):,}")

        split_idx = int((1 - val_split) * len(texts))
        self.train_data = self.tokenizer.tokenize_texts(texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        ds = load_dataset(
            "rojagtap/bookcorpus",
            split="train",
            streaming=True
        )

        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample.get("text", "").strip()
            if isinstance(text, str) and len(text) > 50:
                # --- CLEANUP SPACES BEFORE PUNCTUATION ---
                # Remove space before , . ! ? ; : 
                text = re.sub(r'\s+([,.!?;:])', r'\1', text)

                # Normalize multiple spaces
                text = re.sub(r'\s+', ' ', text)

                # Strip again to remove leading/trailing spaces
                text = text.strip()

                collected.append(text)

        print(f"Collected {len(collected)} samples from Books")
        return collected
