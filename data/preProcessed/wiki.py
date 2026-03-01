# Data/Tiny.py
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer


class WikiDataset:
    """
    TinyStories dataset loader
    Produces a flat token stream for LM training
    """

    def __init__(self, skip=0, take=10000, val_split=0.1,train_mask = None):
        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split
        self.train_mask = train_mask

        print(f"Loading Wiki articles: skip={skip}, take={take}")

        texts = self._load_texts()
        print(f"Total samples: {len(texts)}")
        tokens = self.tokenizer.tokenize_texts(texts)

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx]
        self.val_data = tokens[split_idx:]

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en",
            split="train",
            streaming=True
        )

        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample.get("text", "")
            if not isinstance(text, str):
                continue

            # Split into paragraphs (separated by blank lines)
            paragraphs = text.split("\n\n")

            for paragraph in paragraphs:
                paragraph = paragraph.strip()

                # Count words
                if len(paragraph.split()) > 5:
                    collected.append(paragraph)

        print(f"Collected {len(collected)} paragraphs from Wikipedia")
        return collected
