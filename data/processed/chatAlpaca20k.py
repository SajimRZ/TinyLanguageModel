import json
from itertools import islice

import torch
from datasets import load_dataset

from data.preProcessed.base import BaseTokenizer


class ChatAlpacaDataset:
    def __init__(self, val_split=0.1, take=20000):
        self.tokenizer = BaseTokenizer()
        self.SYSTEM_IDS = self.tokenizer.encode("### System:").tolist()
        self.ASSISTANT_IDS = self.tokenizer.encode("####").tolist()

        texts = self._load_text(take=take)
        print(f"Samples from ChatAlpaca-20K pairs: {len(texts)}")

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

    @staticmethod
    def _extract_pairs(messages):
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                return []

        if not isinstance(messages, list):
            return []

        pairs = []
        for idx in range(len(messages) - 1):
            current_msg = messages[idx]
            next_msg = messages[idx + 1]

            if not isinstance(current_msg, dict) or not isinstance(next_msg, dict):
                continue

            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                user_text = str(current_msg.get("content", "")).strip()
                assistant_text = str(next_msg.get("content", "")).strip()
                if user_text and assistant_text:
                    pairs.append((user_text, assistant_text))

        return pairs

    def _load_text(self, take=20000):
        ds = load_dataset("robinsmits/ChatAlpaca-20K", split="train", streaming=True)
        collected = []

        for sample in islice(ds, take):
            messages = sample.get("messages", [])
            pairs = self._extract_pairs(messages)

            for user_text, assistant_text in pairs:
                system = "### System: You are Lumi. Follow the instruction and respond."
                formatted = f"{system}\n## User: {user_text}\n#### Response: {assistant_text}"
                collected.append(formatted)

        return collected
