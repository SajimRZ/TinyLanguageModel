from itertools import islice

import torch
from datasets import load_dataset

from data.preProcessed.base import BaseTokenizer


class CasualConversationDataset:
    def __init__(self, val_split=0.1, take=50000):
        self.tokenizer = BaseTokenizer()
        self.SYSTEM_IDS = self.tokenizer.encode("### System:").tolist()
        self.ASSISTANT_IDS = self.tokenizer.encode("####").tolist()

        texts = self._load_text(take=take)
        print(f"Samples from casual-conversation: {len(texts)}")

        tokens = self.tokenizer.tokenize_texts(texts)
        print(f"Tokens produced: {len(tokens)}")

        # create mask: 1 = keep, 0 = skip instruction/user context
        mask = torch.ones(len(tokens), dtype=torch.long)
        i = 0
        while i < len(tokens):
            if tokens[i:i + len(self.SYSTEM_IDS)].tolist() == self.SYSTEM_IDS:
                j = i + len(self.SYSTEM_IDS)
                while j < len(tokens) and tokens[j:j + len(self.ASSISTANT_IDS)].tolist() != self.ASSISTANT_IDS:
                    mask[j] = 0
                    j += 1
                i = j
            else:
                i += 1

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx].clone()
        self.val_data = tokens[split_idx:].clone()

        self.train_mask = mask[:split_idx].clone()
        self.val_mask = mask[split_idx:].clone()

        print(f"Training tokens: {self.train_data.numel()}, Validation tokens: {self.val_data.numel()}")
        print(f"Training mask sum: {self.train_mask.sum().item()}, Validation mask sum: {self.val_mask.sum().item()}")

    def _load_text(self, take=50000):
        ds = load_dataset("SohamGhadge/casual-conversation", split="train", streaming=True)
        collected = []

        for sample in islice(ds, take):
            question = str(sample.get("question", "")).strip()
            answer = str(sample.get("answer", "")).strip()

            if not question or not answer:
                continue

            system = "### System: You are Lumi. Follow the instruction and respond."
            formatted = f"{system}\n## User: {question}\n#### Response: {answer}"
            collected.append(formatted)

        return collected
