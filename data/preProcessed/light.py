from pypdf import PdfReader
import re
from typing import List
from pathlib import Path
from data.preProcessed.base import BaseTokenizer


class LightNovelDataset:

    def __init__(
        self,
        series,
        val_split=0.1,
        train_mask=None,
        cache_file="lightnovel_cache.txt",
        append_new=False
    ):
        self.tokenizer = BaseTokenizer()
        self.arr = series
        self.val_split = val_split
        self.texts = []
        self.train_mask = train_mask

        self.cache_path = Path.cwd() / "data" / "processed" / cache_file
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # If cache doesn't exist → build it from scratch
        # -------------------------------------------------
        if not self.cache_path.exists():
            print("Cache not found. Extracting all PDFs...")
            self._extract_and_write(self.arr, mode="w")

        # -------------------------------------------------
        # If append_new=True → append ALL provided PDFs
        # -------------------------------------------------
        elif append_new:
            print("Appending new PDFs to cache...")
            self._extract_and_write(self.arr, mode="a")

        # -------------------------------------------------
        # Always load from cache
        # -------------------------------------------------
        print("Loading text from cache...")
        with open(self.cache_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.texts)} paragraphs from cache.")

        tokens = self.tokenizer.tokenize_texts(self.texts)

        split_idx = int((1 - val_split) * len(tokens))
        self.train_data = tokens[:split_idx]
        self.val_data = tokens[split_idx:]

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    # ==========================================================
    # Extraction + Writing
    # ==========================================================

    def _extract_and_write(self, series_list, mode="w"):
        all_paragraphs = []

        for i in series_list:
            novel = PdfReader(Path.cwd() / "data" / "raw" / i)
            paragraphs = []

            for page in novel.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraph = self.extract_paragraphs(page_text, min_length=30)
                    paragraphs.extend(paragraph)

            paragraphs = paragraphs[4:]  # Skip front matter
            all_paragraphs.extend(paragraphs)

        with open(self.cache_path, mode, encoding="utf-8") as f:
            for para in all_paragraphs:
                f.write(para + "\n")

        print(f"Wrote {len(all_paragraphs)} paragraphs to cache.")

    # ==========================================================
    # Text Cleaning
    # ==========================================================

    def normalize_quotes(self, text: str) -> str:
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
        for fancy, standard in replacements.items():
            text = text.replace(fancy, standard)
        return text

    def extract_paragraphs(self, text: str, min_length: int = 30) -> List[str]:
        text = self.normalize_quotes(text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        paragraphs = re.split(r'\n\s*\n+', text)

        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = ' '.join(para.split())
            if len(cleaned) >= min_length:
                cleaned_paragraphs.append(cleaned)

        return cleaned_paragraphs
