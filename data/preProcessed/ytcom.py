# Data/Tiny.py
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer
import re


class YTcommentsDataset:
    """
    Reddit comments dataset loader
    Produces a cleaned flat token stream for LM training
    """

    def __init__(self, skip=0, take=10000, val_split=0.1):
        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split

        print(f"Loading Youtube Comments: skip={skip}, take={take}")

        texts = self._load_texts()

        print(f"Total usable samples: {len(texts):,}")

        split_idx = int((1 - val_split) * len(texts))
        self.train_data = self.tokenizer.tokenize_texts(texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    # ---------- Cleaning helpers ----------

    def _has_url(self, text: str) -> bool:
        return bool(re.search(r"http[s]?://|www\.", text))

    def _symbol_ratio_too_high(self, text: str, threshold=0.45) -> bool:
        letters = sum(c.isalpha() for c in text)
        symbols = sum(not c.isalnum() and not c.isspace() for c in text)
        if letters == 0:
            return True
        return (symbols / (letters + symbols)) > threshold

    def _normalize_text(self, text: str) -> str:
        text = text.lower()

        # Fix tokenized punctuation spacing
        text = re.sub(r"\s+([?.!,;:])", r"\1", text)
        text = re.sub(r"([?.!,;:])\s+", r"\1 ", text)

        # Fix contractions: it ' s -> it's
        text = re.sub(r"\b(\w+)\s+'\s+(\w+)\b", r"\1'\2", text)

        # Collapse repeated punctuation
        text = re.sub(r"([!?.]){3,}", r"\1\1\1", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ---------- Loader ----------

    def _load_texts(self):
        ds = load_dataset(
            "breadlicker45/youtube-comments-180k",
            split="train",
            streaming=True
        )

        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample.get("body", "")

            if not isinstance(text, str):
                continue

            text = text.strip()

            # Hard filters
            if len(text) < 50:
                continue
            if self._has_url(text):
                continue
            if self._symbol_ratio_too_high(text):
                continue

            text = self._normalize_text(text)

            if len(text) < 25:
                continue

            collected.append(text)

        print(collected)
        return collected
