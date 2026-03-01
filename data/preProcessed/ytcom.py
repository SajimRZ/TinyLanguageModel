from pathlib import Path
from data.preProcessed.base import BaseTokenizer
from pathlib import Path

import random
from pathlib import Path
from data.preProcessed.base import BaseTokenizer


class YTcommentsDataset:
    def __init__(self, val_split=0.1, train_mask=None, seed=42):
        self.tokenizer = BaseTokenizer()
        self.val_split = val_split
        self.train_mask = train_mask

        texts = self._load_texts()
        print(f"total comments from YT: {len(texts)}")

        # ------------------------------
        # RANDOM TEXT-LEVEL SPLIT
        # ------------------------------
        random.seed(seed)
        random.shuffle(texts)

        split_idx = int((1 - val_split) * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        # Tokenize AFTER split (important)
        self.train_data = self.tokenizer.tokenize_texts(train_texts)
        self.val_data = self.tokenizer.tokenize_texts(val_texts)

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        file_path = Path.cwd() / 'data' / 'raw' / 'comments.txt'

        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

        return texts

        