from datasets import load_dataset
from itertools import islice
import re
from data.preProcessed.base import BaseTokenizer


class FourchanDataset:
    def __init__(self, skip, take, val_split=0.1):
        self.QUOTE_PATTERN = re.compile(r"^\s*>+\s?", re.M)
        self.CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
        self.URL_PATTERN = re.compile(r"https?://\S+", re.I)

        self.LEADING_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*", re.M)
        self.OP_PATTERN = re.compile(r"\(OP\)", re.I)
        self.UNKNOWN_PATTERN = re.compile(r"\bunknown\b", re.I)

        self.REPLY_CHAIN_PATTERN = re.compile(r"(>>\d+)+")
        self.INLINE_REPLY_PATTERN = re.compile(r">>+")
        self.GLUED_NUMBER_PATTERN = re.compile(r"^\d+(?=[A-Za-z])")
        self.HEADER_PATTERN = re.compile(
            r"<\|start_header_id\|>.*?<\|end_header_id\|>",
            re.S
        )


        self.tokenizer = BaseTokenizer()
        self.skip = skip
        self.take = take
        self.val_split = val_split

        print(f"Loading 4chan: skip={skip}, take={take}")

        texts = self._load_texts()

        print(f"Total samples of 4chan: {len(texts):,}")

        split_idx = int((1 - val_split) * len(texts))
        self.train_data = self.tokenizer.tokenize_texts(texts[:split_idx])
        self.val_data = self.tokenizer.tokenize_texts(texts[split_idx:])

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")


    def stream_split_clean_group(self,text: str, group_size=8):
        parts = self.HEADER_PATTERN.split(text)
        buffer = []

        for p in parts:
            # drop anything with links
            if self.URL_PATTERN.search(p):
                continue

            p = p.replace("\\", "")
            p = self.CONTROL_PATTERN.sub("", p)

            for line in p.splitlines():
                line = line.strip()
                if not line:
                    continue

                # remove reply chains like >>56>>64
                line = self.REPLY_CHAIN_PATTERN.sub("", line)

                # remove leading numbers (even glued)
                line = self.GLUED_NUMBER_PATTERN.sub("", line)
                line = re.sub(r"^\d+\s*", "", line)

                # remove OP / unknown
                line = self.OP_PATTERN.sub("", line)
                line = self.UNKNOWN_PATTERN.sub("", line)

                # remove leftover reply markers
                line = self.INLINE_REPLY_PATTERN.sub("", line)

                # turn semantic > into commas (lists, comparisons)
                line = re.sub(r"\s*>\s*", ", ", line)

                # normalize whitespace
                line = re.sub(r"\s+", " ", line).strip()

                if len(line) < 10:
                    continue

                buffer.append(line)

                if len(buffer) == group_size:
                    yield buffer
                    buffer = []

        if buffer:
            yield buffer

    def _load_texts(self):

        ds = load_dataset(
            "v2ray/4chan",
            split="train",
            streaming=True
        )
        ds = ds.skip(self.skip)
        collected = []

        for sample in islice(ds, self.take):
            text = sample.get("output", "")
            for group in self.stream_split_clean_group(text):
                st = ""
                for i in group:
                    st += i+"\n"
                collected.append(st)
        return collected