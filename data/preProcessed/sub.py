import pysubs2 as ps
import re
from data.preProcessed.base import BaseTokenizer
from pathlib import Path

class SubtitleDataset:
    """
    Subtitle dataset loader
    Produces a flat token stream for LM training
    """

    def __init__(self, array_of_series, val_split=0.1,train_mask = None):
        self.tokenizer = BaseTokenizer()
        self.arr = array_of_series
        self.val_split = val_split
        self.texts = []

        print(f"Loading subtitles from {len(self.arr)} series...")
        for series in self.arr:
            self.extract_text(series, self.texts)

        print(f"Extracted {len(self.texts)} subtitle lines from {len(self.arr)} entries.")

        # split_idx = int((1 - val_split) * len(self.texts))
        # self.train_data = self.tokenizer.tokenize_texts(self.texts[:split_idx])
        # self.val_data = self.tokenizer.tokenize_texts(self.texts[split_idx:])

        # print(f"Train tokens: {len(self.train_data):,}")
        # print(f"Val tokens: {len(self.val_data):,}")

    def extract_text(self, series, collected):
        cur_dir = Path.cwd()
        parsed = ps.load(cur_dir / "data" / "raw" / series, encoding="utf-8")

        for sub in parsed:
            # Get plaintext (pysubs2 already strips formatting)
            text = sub.plaintext

            # Remove any leftover HTML-like tags just in case
            text = re.sub(r"<[^>]+>", "", text)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Keep only sentences with 3 or more words
            if len(text.split()) >= 3:
                collected.append(text)
