# Data/Tiny.py
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer


class TinyStoriesDataset:
    """
    TinyStories dataset loader
    Produces a flat token stream for LM training
    """

    def __init__(self, skip=0, take=10000, val_split=0.1):
        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split

        print(f"Loading TinyStories: skip={skip}, take={take}")

        texts = self._load_texts()
        print(f"Total samples in Tiny Stories: {len(texts)}")

        split_idx = int((1 - val_split) * len(texts))
        self.train_data = self.tokenizer.tokenize_texts(texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        ds = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=True
        )

        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample.get("text", "")
            if isinstance(text, str) and len(text) > 50:
                collected.append(text)

        print(f"Collected {len(collected)} samples from TinyStories")
        return collected
