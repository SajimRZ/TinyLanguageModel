"""
Flexible Dataset Loader - Load any HuggingFace dataset with custom parameters
"""
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer


class FlexiblLineseDataset:
    """
    Generic dataset loader that works with any HuggingFace dataset
    
    Example usage:
        ds = FlexibleDataset(
            dataset_path="midwestern-simulation-active/ao3_random_subset",
            text_field="text",
            skip=0,
            take=10000,
            split="train",
            val_split=0.1
        )
    """

    def __init__(
        self,
        dataset_path: str,
        text_field: str = "text",
        skip: int = 0,
        take: int = 10000,
        split: str = "train",
        val_split: float = 0.1,
        train_mask=None,
        streaming: bool = True,
        combine_lines: int = 12,
        separator: str = ". ",
        subset: str = None
    ):
        """
        Args:
            dataset_path: HuggingFace dataset path (e.g., "wikitext", "openwebtext")
            text_field: Key name in the dataset for text content (e.g., "text", "content")
            skip: Number of samples to skip
            take: Number of samples to take
            split: Dataset split to use (e.g., "train", "validation")
            val_split: Proportion of data to use for validation (0.0-1.0)
            train_mask: Optional mask for training data
            streaming: Whether to use streaming mode
            combine_lines: Number of lines to combine into one sample
            separator: String to split text into lines
            subset: Optional dataset subset/configuration (e.g., "wikitext-103-v1", "cola")
        """
        self.tokenizer = BaseTokenizer()
        self.dataset_path = dataset_path
        self.text_field = text_field
        self.skip = skip
        self.take = take
        self.split = split
        self.val_split = val_split
        self.train_mask = train_mask
        self.streaming = streaming
        self.combine_lines = combine_lines
        self.separator = separator
        self.subset = subset

        print(f"Loading dataset: {dataset_path}")
        print(f"  Text field: {text_field}")
        print(f"  Skip: {skip}, Take: {take}")
        print(f"  Split: {split}, Streaming: {streaming}")
        if subset:
            print(f"  Subset: {subset}")

        texts = self._load_texts()
        print(f"Total samples: {len(texts)}")
        
        tokens = self.tokenizer.tokenize_texts(texts)

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx]
        self.val_data = tokens[split_idx:]

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        """Load texts from the dataset"""
        load_kwargs = {
            "split": self.split,
            "streaming": self.streaming
        }
        if self.subset:
            load_kwargs["name"] = self.subset
        
        ds = load_dataset(self.dataset_path, **load_kwargs)

        ds = ds.skip(self.skip)
        collected = []
        fallback_fields = ["text", "message", "content", "body"]
        for sample in islice(ds, self.take):
            text = sample.get(self.text_field, "")
            if not isinstance(text, str) or not text.strip():
                for field in fallback_fields:
                    value = sample.get(field, "")
                    if isinstance(value, str) and value.strip():
                        text = value
                        break
            if isinstance(text, str) and len(text) > 25:
                collected.append(text)


        print(f"Collected {len(collected)} samples")
        return collected
