# Data/Tiny.py
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer
from data.preProcessed.sub import SubtitleDataset
import re

class SubtitlesHuggingDataset:


    def __init__(self, skip=0, take=10000, val_split=0.1):
        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split

        print(f"Loading Subs: skip={skip}, take={take}")

        texts = self._load_texts()

        print(f"Total samples of subtitle lines: {len(texts):,}")

        # sub_ds = SubtitleDataset([
        #                       "Horimiya1.srt","Horimiya2.srt","Horimiya3.srt","Horimiya4.srt","Horimiya5.srt","Horimiya6.srt",
        #                       "Horimiya7.srt","Horimiya8.ass","Horimiya9.srt","Horimiya10.ass","Horimiya11.ass","Horimiya12.srt",
        #                       "Horimiya13.srt",
                              
        #                       "Rango.srt","lotr.srt","inception.srt","intersteller.srt","redemption.srt","savingryan.srt","opred.srt",
        #                       "t2.srt","schild.srt", "spiritedaway.srt", "subtitle.srt","pulpfic.srt", "matrix.srt","greenmile.srt",
        #                       "goodhunting.srt","forestgump.srt",
                              
        #                       "mono1.ass","mono2.ass","mono3.ass","mono4.ass","mono5.ass","mono6.ass",
        #                       "mono7.ass","mono8.ass","mono9.ass","mono10.ass"])
        # texts.extend(sub_ds.texts)
        # print(f"Combined total subtitle lines: {len(texts):,}")

        split_idx = int((1 - val_split) * len(texts))
        self.train_data = self.tokenizer.tokenize_texts(texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    def _load_texts(self):
        ds = load_dataset(
            "yhavinga/open_subtitles_en_nl",
            split="train",
            streaming=True
        )

        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample["translation"]["en"]
            if isinstance(text, str) and len(text) > 20:
                # --- CLEANUP SPACES BEFORE PUNCTUATION ---
                # Remove space before , . ! ? ; : 
                text = re.sub(r'\s+([,.!?;:])', r'\1', text)

                # Normalize multiple spaces
                text = re.sub(r'\s+', ' ', text)

                # Strip again to remove leading/trailing spaces
                text = text.strip()

                collected.append(text)

        print(f"Collected {len(collected)} samples from Huggingface subtitles")
        return collected

