from itertools import islice
import re
from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()
from pathlib import Path
import emoji
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False  # if detection fails, discard

def is_valid_utf8(text):
    try:
        text.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False

def clean_text(text):
    # remove emojis
    text = emoji.replace_emoji(text, replace='')
    # remove links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove @mentions and hashtags
    text = re.sub(r'@\S+', '', text)
    # remove timestamps (h:mm:ss or mm:ss)
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
    # collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n', text)
    #colaps repeating strings to a max of 3
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # ignore if fewer than 4 words
    if len(text.split()) < 4:
        return None

    return text
# already done links:
links = [
    
]


cur_path = Path.cwd()
out_path = cur_path / "data" / "raw" / "comments.txt"
sum = 0
for i in range(len(links)):
    count = 0
    existing_comments = set()
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                existing_comments.add(line.strip())
    except FileNotFoundError:
        pass

    # fetch comments
    comments = downloader.get_comments_from_url(
        links[i], 
        sort_by=SORT_BY_POPULAR
    )

    # append cleaned comments
    with open(out_path, "a", encoding="utf-8") as f:
        for comment in comments:
            cleaned = clean_text(comment.get("text", ""))
            if (    cleaned and
                    is_valid_utf8(cleaned) and
                    cleaned not in existing_comments and
                    is_english(cleaned)
                ):
                count += 1
                f.write(cleaned+"\n")
                existing_comments.add(cleaned)
        sum += count
    print(f"done scrapting all comments from {i}: comments found: {count} ")
    print(f"all comments collected: {sum}")