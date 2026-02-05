import torch
import pandas as pd
from data.preProcessed.base import BaseTokenizer
from pathlib import Path
import re
class DailyDialogDataset:
    """Load DailyDialog from local CSVs with proper turn markers"""
    def __init__(self):
        self.tokenizer = BaseTokenizer()
        cur_dir = Path.cwd()
        # Load CSVs
        self.train_df = pd.read_csv(cur_dir/'data'/'raw'/"DDtrain.csv")
        self.val_df = pd.read_csv(cur_dir/'data'/'raw'/"DDvalidation.csv")
        self.test_df = pd.read_csv(cur_dir/'data'/'raw'/"DDtest.csv")

        print(f"Loaded train: {len(self.train_df)} samples")
        print(f"Loaded val: {len(self.val_df)} samples")
        print(f"Loaded test: {len(self.test_df)} samples")

        # Flatten dialogues
        self.train_texts = self._flatten_dialogues(self.train_df)
        self.val_texts = self._flatten_dialogues(self.val_df)
        self.test_texts = self._flatten_dialogues(self.test_df)

        # Tokenize
        self.train_data = self.tokenizer.tokenize_texts(self.train_texts)
        self.val_data = self.tokenizer.tokenize_texts(self.val_texts)
        self.test_data = self.tokenizer.tokenize_texts(self.test_texts)

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")
        print(f"Test tokens: {len(self.test_data):,}")

    def _flatten_dialogues(self, df):
        """Convert dialog column from string of list into text with speakers"""
        if df is None:
            return []
        texts = []
        dialog_column = df['dialog']
        for dialogs in dialog_column:
            for dialog in dialogs.strip("[]").replace("'","").replace('"','').split("\n"):
                # Fix tokenized punctuation spacing
                dialog = re.sub(r"\s+([?.!,;:])", r"\1", dialog)
                dialog = re.sub(r"([?.!,;:])\s+", r"\1 ", dialog)

                # Fix contractions: it ' s -> it's
                dialog = re.sub(r"\b(\w+)\s+'\s+(\w+)\b", r"\1'\2", dialog)

                # Collapse repeated punctuation
                dialog = re.sub(r"([!?.]){3,}", r"\1\1\1", dialog)

                # Collapse whitespace
                dialog = re.sub(r"\s+", " ", dialog).strip()
                dialog = dialog.strip()
                if dialog:
                    texts.append(dialog)

        return texts
