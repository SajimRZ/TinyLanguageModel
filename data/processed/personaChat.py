import json
import re
from itertools import islice

import torch
from datasets import load_dataset

from data.preProcessed.base import BaseTokenizer


class PersonaChatDataset:
    def __init__(self, val_split=0.1, take=50000):
        self.tokenizer = BaseTokenizer()
        self.SYSTEM_IDS = self.tokenizer.encode("### System:").tolist()
        self.ASSISTANT_IDS = self.tokenizer.encode("####").tolist()

        texts = self._load_text(take=take)
        print(f"Samples from persona-chat pairs: {len(texts)}")

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
    def _parse_turn(text):
        if not isinstance(text, str):
            return None, ""

        content = text.strip()
        match = re.match(r"^Persona\s*([AB])\s*:\s*(.*)$", content, flags=re.IGNORECASE)
        if not match:
            return None, ""

        persona = match.group(1).upper()
        body = match.group(2).strip()
        if not body:
            return None, ""

        role = "user" if persona == "A" else "assistant"
        return role, body

    @classmethod
    def _extract_pairs(cls, dialogue):
        if isinstance(dialogue, str):
            try:
                dialogue = json.loads(dialogue)
            except json.JSONDecodeError:
                return []

        if not isinstance(dialogue, list):
            return []

        turns = []
        for line in dialogue:
            role, text = cls._parse_turn(line)
            if role and text:
                turns.append((role, text))

        pairs = []
        for idx in range(len(turns) - 1):
            current_role, current_text = turns[idx]
            next_role, next_text = turns[idx + 1]

            if current_role == "user" and next_role == "assistant":
                pairs.append((current_text, next_text))

        return pairs

    def _load_text(self, take=50000):
        ds = load_dataset("Cynaptics/persona-chat", split="train", streaming=True)
        collected = []

        for sample in islice(ds, take):
            dialogue = sample.get("dialogue", sample.get("dialouge", []))
            pairs = self._extract_pairs(dialogue)

            for user_text, assistant_text in pairs:
                system = "### System: You are Lumi. Follow the instruction and respond."
                formatted = f"{system}\n## User: {user_text}\n#### Response: {assistant_text}"
                collected.append(formatted)

        return collected
