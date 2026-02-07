from itertools import islice
import re
from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()
from pathlib import Path
import emoji


def clean_text(text):
    # remove emojis
    text = emoji.replace_emoji(text, replace='')
    # remove links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove @mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    # remove timestamps (h:mm:ss or mm:ss)
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
    # collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n', text)
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # ignore if fewer than 3 words
    if len(text.split()) < 3:
        return None

    return text

cur_path = Path.cwd()
comments = downloader.get_comments_from_url('https://www.youtube.com/shorts/cs8BErmx4OQ', sort_by=SORT_BY_POPULAR)
out_path = cur_path / "data" / "preProcessed" / "comments.txt"
existing_comments = set()

try:
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            existing_comments.add(line.strip())
except FileNotFoundError:
    pass

with open(out_path, "a", encoding="utf-8") as file:
    for comment in comments:
        cleaned = clean_text(comment.get("text", ""))
        if cleaned and cleaned not in existing_comments:
            file.write(cleaned + "\n")
            existing_comments.add(cleaned)
