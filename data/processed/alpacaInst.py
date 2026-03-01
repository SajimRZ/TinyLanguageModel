import torch
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer

class AlpacaDataset:
    def __init__(self, val_split=0.1):
        self.tokenizer = BaseTokenizer()
        self.SYSTEM_IDS = self.tokenizer.encode("### System:")
        self.ASSISTANT_IDS = self.tokenizer.encode("####")
        
        texts = self._load_text()
        print(f"Samples from Alpaca: {len(texts)}")
        
        tokens = self.tokenizer.tokenize_texts(texts)
        print(f"Tokens produced: {len(tokens)}")
        
        # create mask: 1 = keep, 0 = skip instruction
        mask = torch.ones(len(tokens), dtype=torch.long)
        i = 0
        while i < len(tokens):
            # detect system start
            if tokens[i:i+len(self.SYSTEM_IDS)].tolist() == self.SYSTEM_IDS:
                # skip instruction until assistant response
                j = i + len(self.SYSTEM_IDS)
                while j < len(tokens) and tokens[j:j+len(self.ASSISTANT_IDS)].tolist() != self.ASSISTANT_IDS:
                    mask[j] = 0
                    j += 1
                i = j  # continue from assistant token
            else:
                i += 1
        
        # split
        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx].clone()
        self.val_data = tokens[split_idx:].clone()
        
        self.train_mask = mask[:split_idx].clone()
        self.val_mask = mask[split_idx:].clone()
        
        print(f"Training tokens: {self.train_data.numel()}, Validation tokens: {self.val_data.numel()}")
        print(f"Training mask sum: {self.train_mask.sum().item()}, Validation mask sum: {self.val_mask.sum().item()}")
    
    def _load_text(self):
        from datasets import load_dataset
        from itertools import islice
        
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        collected = []
        for sample in islice(ds, 51000):
            system = "### System: You are Lumi. Follow the instruction and respond."
            instruction = sample.get("instruction", "")
            input_given = sample.get("input", "")
            if input_given != "":
                input_given = ": " + input_given
            text = sample.get("output", "")
            formatted = f"{system}\n## User: {instruction}{input_given}\n#### Response: {text}"
            collected.append(formatted)
        return collected