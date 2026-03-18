import torch
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer


class PersonaChatDataset:
    def __init__(self, val_split=0.1, max_samples=20000):
        self.tokenizer = BaseTokenizer()
        self.system_prefix = "### System: "
        self.user_prefix = "### User: "
        self.assistant_prefix = "### Assistant: "
        self.system_message = "You are Lumi. Follow the conversation and respond."

        samples = self._load_samples(max_samples=max_samples)
        print(f"Samples from Cynaptics/persona-chat: {len(samples)}")

        tokens, mask = self._tokenize_with_mask(samples)
        print(f"Tokens produced: {len(tokens)}")

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx].clone()
        self.val_data = tokens[split_idx:].clone()

        self.train_mask = mask[:split_idx].clone()
        self.val_mask = mask[split_idx:].clone()

        print(
            f"Training tokens: {self.train_data.numel()}, "
            f"Validation tokens: {self.val_data.numel()}"
        )
        print(
            f"Training mask sum: {self.train_mask.sum().item()}, "
            f"Validation mask sum: {self.val_mask.sum().item()}"
        )

    def _build_segments(self, sample):
        dialogue = sample.get("dialogue", []) or []
        reference = str(sample.get("reference", "")).strip()

        segments = [("system", f"{self.system_prefix}{self.system_message}\n")]

        # dialogue alternates A/B where A maps to user and B maps to assistant
        for turn_idx, turn in enumerate(dialogue):
            text = str(turn).strip()
            if not text:
                continue

            if turn_idx % 2 == 0:
                segments.append(("user", f"{self.user_prefix}{text}\n"))
            else:
                segments.append(("assistant", f"{self.assistant_prefix}{text}\n"))

        # dataset provides a persona-B target response for the final dialogue context
        if reference:
            segments.append(("assistant", f"{self.assistant_prefix}{reference}\n"))

        return segments

    def _load_samples(self, max_samples):
        ds = load_dataset("Cynaptics/persona-chat", split="train", streaming=True)
        collected = []
        for sample in islice(ds, max_samples):
            collected.append(self._build_segments(sample))
        return collected

    def _tokenize_with_mask(self, samples):
        token_buffer = []
        mask_buffer = []

        for segments in samples:
            for role, text in segments:
                ids = self.tokenizer.enc.encode(text)
                token_buffer.extend(ids)

                # Train only on assistant turns (including assistant prefix).
                if role == "assistant":
                    mask_buffer.extend([1] * len(ids))
                else:
                    mask_buffer.extend([0] * len(ids))

            token_buffer.append(self.tokenizer.eot_token)
            mask_buffer.append(0)

        tokens = torch.tensor(token_buffer, dtype=torch.long)
        mask = torch.tensor(mask_buffer, dtype=torch.long)
        return tokens, mask


# Backward-compatible alias used by existing training imports.
AlpacaDataset = PersonaChatDataset