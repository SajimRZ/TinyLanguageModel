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
        self.train_mask = train_mask

        print(f"Loading Books: skip={skip}, take={take}")

        texts = self._load_texts()

        print(f"Total samples of book lines: {len(texts):,}")

        tokens = self.tokenizer.tokenize_texts(texts)

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx]
        self.val_data = tokens[split_idx:]

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
        buffer = []

        for sample in islice(ds, self.take):
            text = sample.get("text", "").strip()

            if isinstance(text, str):

                # --- CLEANUP SPACES BEFORE PUNCTUATION ---
                text = re.sub(r'\s+([,.!?;:])', r'\1', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()

                buffer.append(text)

                # When we have 10 texts → merge them
                if len(buffer) == 10:
                    combined = " ".join(buffer)
                    collected.append(combined)
                    buffer = []

        # Handle leftover texts (if total not divisible by 5)
        if buffer:
            combined = " ".join(buffer)
            collected.append(combined)

        print(f"Collected {len(collected)} combined samples from Books")
        return collected
