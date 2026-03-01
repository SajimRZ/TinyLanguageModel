import json
import re
from datasets import load_dataset
from itertools import islice
from data.preProcessed.base import BaseTokenizer


class FourchanDataset:
	"""
	Fal7acy 4chan archive dataset loader
	Produces a flat token stream for LM training
	Uses only `content` values from the `posts` column
	"""

	def __init__(self, skip=0, take=10000, val_split=0.1, train_mask=None):
		self.tokenizer = BaseTokenizer()
		self.skip = skip
		self.take = take
		self.val_split = val_split
		self.train_mask = train_mask

		print(f"Loading Fal7acy_4chan_archive_JSONL: skip={skip}, take={take}")

		texts = self._load_texts()
		self.texts = texts
		print(f"Total extracted content lines: {len(texts)}")
		tokens = self.tokenizer.tokenize_texts(texts)

		split_idx = int((1 - val_split) * len(tokens))
		self.train_data = tokens[:split_idx]
		self.val_data = tokens[split_idx:]

		print(f"Train tokens: {len(self.train_data):,}")
		print(f"Val tokens: {len(self.val_data):,}")

	def _parse_post_item(self, item):
		def _clean_content(text):
			cleaned_lines = []

			for line in text.splitlines():
				line = re.sub(r">>\d+\b", "", line)
				line = re.sub(r"^\s*>>\s*", "", line)
				line = re.sub(r"^\s*>\s*", "", line)
				line = line.strip()
				if line:
					cleaned_lines.append(line)

			if not cleaned_lines:
				return None

			return "\n".join(cleaned_lines)

		if isinstance(item, dict):
			content = item.get("content", "")
			if isinstance(content, str):
				content = _clean_content(content)
				return content if content else None

		if isinstance(item, str):
			line = item.strip()
			if not line:
				return None

			try:
				parsed = json.loads(line)
				if isinstance(parsed, dict):
					content = parsed.get("content", "")
					if isinstance(content, str):
						content = _clean_content(content)
						return content if content else None
			except json.JSONDecodeError:
				return None

		return None

	def _extract_contents_from_posts(self, posts):
		contents = []

		if isinstance(posts, list):
			for item in posts:
				content = self._parse_post_item(item)
				if content:
					contents.append(content)
			return contents

		if isinstance(posts, str):
			raw = posts.strip()
			if not raw:
				return contents

			try:
				parsed = json.loads(raw)
				if isinstance(parsed, list):
					for item in parsed:
						content = self._parse_post_item(item)
						if content:
							contents.append(content)
					return contents
				if isinstance(parsed, dict):
					content = self._parse_post_item(parsed)
					if content:
						contents.append(content)
					return contents
			except json.JSONDecodeError:
				pass

			for line in raw.splitlines():
				content = self._parse_post_item(line)
				if content:
					contents.append(content)

		return contents

	def _load_texts(self):
		ds = load_dataset(
			"adamo1139/Fal7acy_4chan_archive_JSONL",
			split="train",
			streaming=True,
		)

		ds = ds.skip(self.skip)
		collected = []

		for sample in islice(ds, self.take):
			posts = sample.get("posts", None)
			if posts is None:
				continue

			collected.extend(self._extract_contents_from_posts(posts))

		print(f"Collected {len(collected)} content lines from Fal7acy 4chan archive")
		return collected


if __name__ == "__main__":
	loader = FourchanDataset(skip=0, take=10)
	preview = loader.texts[:5]

	print("\nFirst 5 extracted content lines:")
	for idx, text in enumerate(preview, start=1):
		print(f"{idx}. {text}")
